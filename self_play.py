import math
import time

import numpy
import ray
import torch

import models

import discord_io


@ray.remote
class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """
    # if worker_id == -1: test_play, else: self_play
    def __init__(self, initial_checkpoint, Game, config, seed, worker_id):
        self.config = config
        self.game = Game(seed)

        # Fix random generator seed
        numpy.random.seed(seed)
        torch.manual_seed(seed)

        # Initialize the network
        self.model = models.MuZeroNetwork(self.config)
        self.model.set_weights(initial_checkpoint["weights"])
        self.model.to(torch.device("cuda" if self.config.selfplay_on_gpu else "cpu"))
        self.model.eval()

        # CHANGED ---------------------------------------------------------------
        self.worker_id = worker_id
        if worker_id == -1:
            discord_io.test_play_send("Initialized!")
        else:
            discord_io.self_play_send(seed, "Initialized!")
        # -----------------------------------------------------------------------

    def continuous_self_play(self, shared_storage, replay_buffer, test_mode=False):
        while ray.get(
            shared_storage.get_info.remote("training_step")
        ) < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            self.model.set_weights(ray.get(shared_storage.get_info.remote("weights")))

            # CHANGED---------------------------------------------------------------------
            io_training_step = ray.get(shared_storage.get_info.remote("training_step"))
            # ----------------------------------------------------------------------------

            if not test_mode:
                
                game_history = self.play_game(
                    temperature=self.config.visit_softmax_temperature_fn(
                        trained_steps=io_training_step # changed
                    ),
                    temperature_threshold=self.config.temperature_threshold,
                    render=True,
                    opponent="self",
                    muzero_player=0,
                )
                print('finished play_game.')
                replay_buffer.save_game.remote(game_history, shared_storage)
                print('finished save_game.')
                # CHANGED---------------------------------------------------------------------
                discord_io.self_play_send(self.worker_id, "Trained step: {}\nSelf-played step: {}\nEpisode have finished!".format(
                    io_training_step,
                    ray.get(shared_storage.get_info.remote("num_played_games")),
                ))
                # ----------------------------------------------------------------------------
            else:
                # Take the best action (no exploration) in test mode
                game_history = self.play_game(
                    temperature=0,
                    temperature_threshold=self.config.temperature_threshold,
                    render=False,
                    opponent="self" if len(self.config.players) == 1 else self.config.opponent,
                    muzero_player=self.config.muzero_player,
                )

                # Save to the shared storage
                shared_storage.set_info.remote(
                    {
                        "episode_length": len(game_history.action_history) - 1,
                        "total_reward": sum(game_history.reward_history),
                        "mean_value": numpy.mean(
                            [value for value in game_history.root_values if value]
                        ),
                    }
                )

                # CHANGED---------------------------------------------------------------------
                io_muzero_outcome = "None"
                # ----------------------------------------------------------------------------
                
                if 1 < len(self.config.players):
                    # CHANGED-----------------------------------------------------------------
                    io_muzero_reward = sum(
                        reward
                        for i, reward in enumerate(game_history.reward_history)
                        if game_history.to_play_history[i - 1]
                        == self.config.muzero_player
                    )
                    io_opponent_reward = sum(
                        reward
                        for i, reward in enumerate(game_history.reward_history)
                        if game_history.to_play_history[i - 1]
                        != self.config.muzero_player
                    )
                    # ------------------------------------------------------------------------

                    shared_storage.set_info.remote(
                        {
                            "muzero_reward": io_muzero_reward, # changed
                            "opponent_reward": io_opponent_reward, # changed
                        }
                    )

                    # CHANGED----------------------------------------------------------------
                    io_muzero_outcome = "WIN" if io_muzero_reward > 0 else "LOSE" if io_opponent_reward > 0 else "DRAW"
                    # -----------------------------------------------------------------------

                
                # CHANGED---------------------------------------------------------------------
                discord_io.test_play_send("Trained step: {}\nSelf-played step: {}\nEpisode have finished!\nMuZero Player {}!".format(
                    io_training_step,
                    ray.get(shared_storage.get_info.remote("num_played_games")),
                    io_muzero_outcome,
                ))
                # ----------------------------------------------------------------------------

            # Managing the self-play / training ratio
            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)
            if not test_mode and self.config.ratio:
                while (
                    ray.get(shared_storage.get_info.remote("training_step"))
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    < self.config.ratio
                    and ray.get(shared_storage.get_info.remote("training_step"))
                    < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

        self.close_game(shared_storage)

    def play_game(
        self, temperature, temperature_threshold, render, opponent, muzero_player
    ):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        with torch.no_grad():
            while (
                not done and len(game_history.action_history) <= self.config.max_moves
            ):
                assert (
                    len(numpy.array(observation).shape) == 3
                ), f"Observation should be 3 dimensionnal instead of {len(numpy.array(observation).shape)} dimensionnal. Got observation of shape: {numpy.array(observation).shape}"
                assert (
                    numpy.array(observation).shape == self.config.observation_shape
                ), f"Observation should match the observation_shape defined in MuZeroConfig. Expected {self.config.observation_shape} but got {numpy.array(observation).shape}."
                stacked_observations = game_history.get_stacked_observations(
                    -1, self.config.stacked_observations, len(self.config.action_space)
                )

                # Choose the action
                if opponent == "self" or muzero_player == self.game.to_play():
                    legal_actions = self.game.legal_actions()
                    root, mcts_info = MCTS(self.config).run(
                        self.model,
                        stacked_observations,
                        legal_actions,
                        self.game.to_play(),
                        True,
                    )
                    action = self.select_action(
                        root,
                        temperature
                        if not temperature_threshold
                        or len(game_history.action_history) < temperature_threshold
                        else 0,
                    )

                    if render:
                        print(f'game_history count: {len(game_history.action_history)}')
                        print(f'legal_actions: {legal_actions}')
                        print(f'action: {action}')
                        print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                        print(
                            f"Root value for player {self.game.to_play()}: {root.value():.2f}"
                        )
                else:
                    action, root = self.select_opponent_action(
                        opponent, stacked_observations
                    )

                observation, reward, done = self.game.step(action)

                if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

                game_history.store_search_statistics(root, self.config.action_space)

                # Next batch
                game_history.action_history.append(action)
                game_history.observation_history.append(observation)
                game_history.reward_history.append(reward)
                game_history.to_play_history.append(self.game.to_play())

        return game_history

    def close_game(self, shared_storage):
        self.game.close()
        print("shutdown self_play.....")
        if self.worker_id == -1:
            shared_storage.set_info.remote("terminated_test_play", True)
        else:
            shared_storage.set_list_info.remote("terminated_self_play", self.worker_id, True)


    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            print(f'Tree depth: {mcts_info["max_tree_depth"]}')
            print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")
            print(
                f"Player {self.game.to_play()} turn. MuZero suggests {self.game.action_to_string(self.select_action(root, 0))}"
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            assert (
                self.game.legal_actions()
            ), f"Legal actions should not be an empty array. Got {self.game.legal_actions()}."
            assert set(self.game.legal_actions()).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @staticmethod
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = numpy.array(
            [child.visit_count for child in node.children.values()], dtype="int32"
        )
        actions = [action for action in node.children.keys()]
        if temperature == 0:
            action = actions[numpy.argmax(visit_counts)]
        elif temperature == float("inf"):
            action = numpy.random.choice(actions)
        else:
            # See paper appendix Data Generation
            visit_count_distribution = visit_counts ** (1 / temperature)
            visit_count_distribution = visit_count_distribution / sum(
                visit_count_distribution
            )
            action = numpy.random.choice(actions, p=visit_count_distribution)

        return action


# Game independent
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(
        self,
        model,
        observation,
        legal_actions,
        to_play,
        add_exploration_noise,
        override_root_with=None,
    ):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.

        tree の root で、representation function を使用して、current observation から hidden state を取得します。
        次に、action sequence と network によって学習された model のみを使用して、MCTS を実行します。
        """
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = ChoiceNode(0)
            observation = (
                torch.tensor(observation)
                .float()
                .unsqueeze(0)
                .to(next(model.parameters()).device)
            )
            # initial_inference は representation と prediction を含む
            (
                root_predicted_value, # vector (1, 1), domain:[0, 1] (tensor type)
                reward, # vector (1, 1), value:0 (tensor type)
                policy_logits, # vector (1, config.action_space), domain: [NOT YET SOFTMAXED] (tensor type)
                hidden_state,
                choice_logits, # vector (1, config.num_choice), domain: [NOT YET SOFTMAXED] (tensor type) # ADDED -------------------------------------------------------------------
            ) = model.initial_inference(observation)
            # FIXED --------------------------------------------------------------------------------------
            root_predicted_value = root_predicted_value[0,0].item() # tensor to scalar
            reward = reward[0,0].item() # tensor to scalar
            # --------------------------------------------------------------------------------------------
            
            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."
            root.expand(
                to_play,
                reward,
                legal_actions,
                policy_logits,
                hidden_state,
                self.config.num_choice,
                choice_logits,
            )
        # add exploration_noise for noise
        # do not add noise for choice
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            # virtual_to_play = to_play
            choice_node = root
            search_path_choiced_node = [choice_node]
            search_path_actioned_node = []
            current_tree_depth = 0

            while choice_node.expanded():
                current_tree_depth += 1
                action, action_node = self.select_action_child(choice_node, min_max_stats)
                search_path_actioned_node.append(action_node)
                choice, choiced_node = self.select_choice_child(action_node)
                search_path_choiced_node.append(choiced_node)

                # Players play turn by turn
                #if virtual_to_play + 1 < len(self.config.players):
                #    virtual_to_play = self.config.players[virtual_to_play + 1]
                #else:
                #    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent_choice = search_path_choiced_node[-2]
            # recurrent_interface は gynamics と prediction を含む
            # value, reward, policy_logits, next_encoded_state, choice_logits, player
            ( 
                value, 
                reward, 
                policy_logits, 
                hidden_state, 
                choice_logits, 
                to_play_logits
            ) = model.recurrent_inference(
                parent_choice.hidden_state,
                torch.tensor([[action]]).to(parent_choice.hidden_state.device),
                torch.tensor([[choice]]).to(parent_choice.hidden_state.device)
            )
            # policy_logits.shape: (1,29)
            # FIXED ----------------------------------------------------------------------------------
            value = value[0,0].item() # scalar [0,1]
            reward = reward[0,0].item()

            ## ADDED -----------------------------------------------------------------------------------
            ## PLAYER CATEGORIZE
            virtual_to_play = to_play_logits[0,0].item() # scalar [0, 1]
            virtual_to_play = 1 if virtual_to_play > 0.5 else 0
            ## ----------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------

            choice_node.expand(
                virtual_to_play,
                reward,
                self.config.action_space,
                policy_logits,
                hidden_state,
                self.config.num_choice,
                choice_logits,
            )

            self.backpropagate(search_path_actioned_node, search_path_choiced_node, value, virtual_to_play, min_max_stats)

            max_tree_depth = max(max_tree_depth, current_tree_depth)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info
    
    def select_choice_child(self, action_node):
        """
        Select the choice child
        """
        max_score = max(
            self.quasi_random_score(choice_child)
            for choice_child in action_node.children.items()
        )
        choice = numpy.random.choice(
            [
                choice
                for choice, child_choice in action_node.children.items()
                if self.quasi_random_score(child_choice) == max_score
            ]
        )
        return choice, action_node.children[choice]
    
    def quasi_random_score(self, child_choice):
        return child_choice.choice_prob / (child_choice.visit_count + 1)

    def select_action_child(self, choice_node, min_max_stats):
        """
        Select the action child with the highest UCB score.
        """
        max_ucb = max(
            self.ucb_score(choice_node, action_child, min_max_stats)
            for action_child in choice_node.children.items()
        )
        assert len([
                action
                for action, action_child in choice_node.children.items()
                if self.ucb_score(choice_node, action_child, min_max_stats) == max_ucb
                ]) > 0, f"node.children.items(): {[action for action, _ in choice_node.children.items()]}\n{[child for _, child in choice_node.children.items()]}"
        action = numpy.random.choice(
            [
                action
                for action, child in choice_node.children.items()
                if self.ucb_score(choice_node, child, min_max_stats) == max_ucb
            ]
        )
        return action, choice_node.children[action]

    def ucb_score(self, parent_choice, child_action, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent_choice.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent_choice.visit_count) / (child_action.visit_count + 1)

        prior_score = pb_c * child_action.prior

        if child_action.visit_count > 0:
            # Mean value Q
            value_score = min_max_stats.normalize(
                child_action.reward() 
                + self.config.discount
                # * (child_action.value() if len(self.config.players) == 1 else -child_action.value())
                * child_action.value()
            )
        else:
            value_score = 0
        assert prior_score + value_score or prior_score + value_score == 0, f"ASSERT UCB_SCORE\n{prior_score}\n{value_score}"

        return prior_score + value_score

    def backpropagate(self, search_path_actioned_node, search_path_choiced_node, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if len(self.config.players) == 1:
            for actioned_node in reversed(search_path_actioned_node):
                actioned_node.value_sum += value
                actioned_node.visit_count += 1
                min_max_stats.update(actioned_node.reward() + self.config.discount * actioned_node.value())

                value = actioned_node.reward() + self.config.discount * value

            for choiced_node in reversed(search_path_choiced_node):
                choiced_node.visit_count += 1

        elif len(self.config.players) == 2:
            for actioned_node in reversed(search_path_actioned_node):
                actioned_node.value_sum += value if actioned_node.to_played == to_play else -value
                actioned_node.visit_count += 1
                min_max_stats.update(actioned_node.reward() + self.config.discount * -actioned_node.value())

                # node内のrewardは親ノードのアクションによって出力される。node内のto_playではない方の報酬である
                # 確率的MuZeroの場合は違う。rewardとto_playedは一致している。
                # このプログラムはゼロサムゲームのシミュレーションであるからそもそもrewardは必要ないのだが。
                value = (
                    actioned_node.reward() if actioned_node.to_played == to_play else -actioned_node.reward()
                ) + self.config.discount * value
                
            for choiced_node in reversed(search_path_choiced_node):
                choiced_node.visit_count += 1

        else:
            raise NotImplementedError("More than two player mode not implemented.")


class Node:
    def __init__(self, prior, to_played):
        self.to_played = to_played
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def reward(self):
        reward_sum = 0
        expanded_choice_node_count = 0
        for choice_node in self.children:
            if choice_node.expanded():
                expanded_choice_node_count += 1
                reward_sum += choice_node.reward
        if expanded_choice_node_count == 0:
            return 0
        else:
            return reward_sum / expanded_choice_node_count

    def expand(self, num_choices, choices_logits):
        """
        We expand a node using the value, reward and policy prediction obtained from the
        neural network.
        """
        choice_values = torch.softmax(
            torch.tensor([choices_logits[0][a] for a in range(num_choices)]), dim=0
        )
        choices = {c: choice_values[i] for i, c in enumerate(range(num_choices))} # ADDED ---------------------
        for choice, c_prob in choices.items():
            self.children[choice] = ChoiceNode(c_prob)

class ChoiceNode:
    def __init__(self, choice_prob):
        self.visit_count = 0
        self.choice_prob = choice_prob
        self.hidden_state = None
        self.children = {}
        self.reward = 0
        self.to_play = -1
    
    def expanded(self):
        return self.children[0].expanded()
    
    def expand(self, to_play, reward, actions, policy_logits, hidden_state, num_choices, choices_logits):
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = torch.softmax(
            torch.tensor([policy_logits[0][a] for a in actions]), dim=0
        ).tolist()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, p in policy.items():
            self.children[action] = Node(p, to_play)
            self.children[action].expand(num_choices, choices_logits)
    
    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac

class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []
        self.reanalysed_predicted_root_values = None
        # For PER
        self.priorities = None
        self.game_priority = None

    def store_search_statistics(self, root, action_space):
        # Turn visit count from root into a policy
        if root is not None:
            sum_visits = sum(child.visit_count for child in root.children.values())
            self.child_visits.append(
                [
                    root.children[a].visit_count / sum_visits
                    if a in root.children
                    else 0
                    for a in action_space
                ]
            )

            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(
        self, index, num_stacked_observations, action_space_size
    ):
        """
        Generate a new observation with the observation at the index position
        and num_stacked_observations past observations and actions stacked.
        """
        # Convert to positive index
        index = index % len(self.observation_history)

        stacked_observations = self.observation_history[index].copy()
        for past_observation_index in reversed(
            range(index - num_stacked_observations, index)
        ):
            if 0 <= past_observation_index:
                previous_observation = numpy.concatenate(
                    (
                        self.observation_history[past_observation_index],
                        [
                            numpy.ones_like(stacked_observations[0])
                            * self.action_history[past_observation_index + 1]
                            / action_space_size
                        ],
                    )
                )
            else:
                previous_observation = numpy.concatenate(
                    (
                        numpy.zeros_like(self.observation_history[index]),
                        [numpy.zeros_like(stacked_observations[0])],
                    )
                )

            stacked_observations = numpy.concatenate(
                (stacked_observations, previous_observation)
            )

        return stacked_observations


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value
