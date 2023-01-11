import copy
import time

import numpy
import ray
import torch

import models

import discord_io


@ray.remote
class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_checkpoint, config):
        self.config = config

        # Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(torch.device("cuda" if self.config.train_on_gpu else "cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        if "cuda" not in str(next(self.model.parameters()).device):
            print("You are not training on GPU.\n")

        # Initialize the optimizer
        if self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr_init,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.lr_init,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise NotImplementedError(
                f"{self.config.optimizer} is not implemented. You can change the optimizer manually in trainer.py."
            )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )
        
        # CHANGED ------------------------------------------------------------
        discord_io.trainer_send("Initialized!")
        # --------------------------------------------------------------------

    def continuous_update_weights(self, replay_buffer, shared_storage):
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(1)

        next_batch = replay_buffer.get_batch.remote()
        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(next_batch)
            next_batch = replay_buffer.get_batch.remote()
            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
                choice_loss,
                to_play_loss
            ) = self.update_weights(batch)

            # CHANGED ------------------------------------------------------------
            # discord_io.trainer_send("[{}/{}] Updated weights!".format(self.training_step, self.config.training_steps))
            #---------------------------------------------------------------------

            if self.config.PER:
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": copy.deepcopy(
                            models.dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )
                if self.config.save_model:
                    # CHANGED ------------------------------------------------------------
                    discord_io.trainer_send("[{}/{}] Saved Checkpoint!".format(self.training_step, self.config.training_steps))
                    #---------------------------------------------------------------------
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                    "choice_loss": choice_loss,
                    "to_play_loss": to_play_loss
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)
        print("shutdown trainer...")
        shared_storage.set_info.remote("terminated_trainer", True)

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch_numpy,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            target_to_play,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        # [PAPER INFO] FOR BOARD GAMES, STATE ARE SAMPLED UNIFORMLY.
        # FIXED ----------------------------------------------------------------------------
        #target_value_scalar = numpy.array(target_value, dtype="float32")
        #priorities = numpy.zeros_like(target_value_scalar)
        # ----------------------------------------------------------------------------------
        
        device = next(self.model.parameters()).device
        if self.config.PER:
            weight_batch = torch.tensor(weight_batch.copy()).float().to(device)

        observation_batch = torch.tensor(observation_batch_numpy).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)
        target_to_play = torch.tensor(target_to_play).float().to(device)
        gradient_scale_batch = torch.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: num_unroll_steps+1, batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # target_to_play: batch, num_unroll_steps+1
        # gradient_scale_batch: batch, num_unroll_steps+1

        ## representation -> prediction
        value, reward, policy_logits, hidden_state, choice_logits = self.model.initial_inference(
            observation_batch[0]
        )

        # value: batch, 1
        # reward: batch, 1
        # policy_logits: batch, len(action_space)
        # hidden_state: batch, channels(in the ResNet), height, width

        to_play = target_to_play[:, 0] # to_play: batch, 1
        
        predictions = [(value, reward, policy_logits, choice_logits, to_play)]
        target_choice = torch.zeros(self.config.batch_size, action_batch.shape[1], self.config.num_choice)
        for i in range(1, action_batch.shape[1]): # num_unroll_steps
            target_hidden_state = self.model.representation(observation_batch[i])
            (
                adopted_hidden_state, # batch, channels(in the ResNet), height, width
                adopted_reward, # batch, 1
                adopted_to_play, # batch, 1
            ) = self.model.dynamics(hidden_state, action_batch[:, i], 0)
            adopted_loss = ((target_hidden_state - adopted_hidden_state)**2).mean(dim=(1,2,3)) # batch
            for choice in range(1, self.config.num_choice):
                (
                    temp_hidden_state, # batch, channels(in the ResNet), height, width
                    temp_reward, # batch, 1
                    temp_to_play # batch, 1
                ) = self.model.dynamics(hidden_state, action_batch[:, i], choice)
                temp_loss = ((target_hidden_state - temp_hidden_state)**2).mean(dim=(1,2,3)) # batch
                
                for batch in range(self.config.batch_size):
                    if temp_loss[batch] < adopted_loss[batch]:
                        adopted_loss[batch] = temp_loss[batch]
                        adopted_hidden_state[batch] = temp_hidden_state[batch]
                        adopted_reward[batch] = temp_reward[batch]
                        adopted_to_play[batch] = temp_to_play[batch]
                        target_choice[batch, i, choice] = 1
            hidden_state = adopted_hidden_state
            reward = adopted_reward
            to_play = adopted_to_play
            value, policy_logits, choice_logits = self.model.prediction(hidden_state)

            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy_logits, choice_logits, to_play))

        ## Compute losses
        value_loss, reward_loss, policy_loss, choice_loss, to_play_loss = (0, 0, 0, 0, 0)
        value, reward, policy_logits, choice_logits, to_play = predictions[0]
        # Ignore reward loss for the first batch step
        (
            current_value_loss,
            current_reward_loss,
            current_policy_loss,
            current_choice_loss,
            current_to_play_loss
        ) = self.loss_function(
            value.squeeze(-1), # batch
            reward.squeeze(-1), # batch
            policy_logits,  # batch, action_space
            choice_logits, # batch, choice_num
            to_play, # batch
            target_value[:, 0], # batch
            target_reward[:, 0], # batch
            target_policy[:, 0], # batch, action_space
            target_choice[:, 0], # batch, num_choice
            target_to_play[:, 0] # batch
        )
        value_loss += current_value_loss
        reward_loss += current_reward_loss
        policy_loss += current_policy_loss
        choice_loss += current_choice_loss
        to_play_loss += current_to_play_loss

        # Compute priorities for the prioritized replay (See paper appendix Training)
        # [PAPER INFO] FOR BOARD GAMES, STATE ARE SAMPLED UNIFORMLY.
        # FIXED ----------------------------------------------------------------------------
        #pred_value_scalar = (
        #    models.support_to_scalar(value, self.config.support_size)
        #    .detach()
        #    .cpu()
        #    .numpy()
        #    .squeeze()
        #)
        #priorities[:, 0] = (
        #    numpy.abs(pred_value_scalar - target_value_scalar[:, 0])
        #    ** self.config.PER_alpha
        #)
        # -----------------------------------------------------------------------------------

        for i in range(1, len(predictions)):
            value, reward, policy_logits, choice_logits, to_play = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
                current_choice_loss,
                current_to_play_loss
            ) = self.loss_function(
                value.squeeze(-1), # batch
                reward.squeeze(-1), # batch
                policy_logits,  # batch, action_space
                choice_logits, # batch, choice_num
                to_play, # batch
                target_value[:, i], # batch
                target_reward[:, i], # batch
                target_policy[:, i], # batch, action_space
                target_choice[:, i], # batch, num_choice
                target_to_play[:, i] # batch
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            # FIXED -----------------------------------------------------------------------------------
            #current_reward_loss.register_hook(
            #    lambda grad: grad / gradient_scale_batch[:, i]
            #)
            # -----------------------------------------------------------------------------------------
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_choice_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_to_play_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss # 0
            policy_loss += current_policy_loss
            choice_loss += current_choice_loss
            to_play_loss += current_to_play_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            # [PAPER INFO] FOR BOARD GAMES, STATE ARE SAMPLED UNIFORMLY.
            # FIXED -------------------------------------------------------------------------------------
            #pred_value_scalar = (
            #    models.support_to_scalar(value, self.config.support_size)
            #    .detach()
            #    .cpu()
            #    .numpy()
            #    .squeeze()
            #)
            #priorities[:, i] = (
            #    numpy.abs(pred_value_scalar - target_value_scalar[:, i])
            #    ** self.config.PER_alpha
            #)
            # -------------------------------------------------------------------------------------------

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss + choice_loss + to_play_loss
        if self.config.PER:
            # Correct PER bias by using importance-sampling (IS) weights
            loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize 重みの更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            None, #priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
            choice_loss.mean().item(),
            to_play_loss.mean().item()
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value, # batch
        reward, # batch
        policy_logits, # batch, action_space
        choice_logits, # batch, num_choice
        to_play, # batch
        target_value, # batch
        target_reward, # batch
        target_policy, # batch, action_space
        target_choice, # batch, num_choice
        target_to_play # batch
    ):
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (value - target_value)**2
        reward_loss = reward # all zero
        policy_loss = (-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits)).sum(1)
        choice_loss = (-target_choice * torch.nn.LogSoftmax(dim=1)(choice_logits)).sum(1)
        to_play_loss = (to_play - target_to_play)**2
        return value_loss, reward_loss, policy_loss, choice_loss, to_play_loss
