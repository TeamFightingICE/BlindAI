from loguru import logger


class CollectDataHelper:
    total_round_data = []
    total_round_action_data = []
    total_round_action_dist_data = []
    total_round_actor_hidden_data = []

    current_round_data = []
    current_round_action = []
    current_round_action_dist_data = []
    current_round_actor_hidden_data = []

    def __init__(self) -> None:
        self.total_round_data = []
        self.total_round_action_data = []
        self.total_round_action_dist_data = []
        self.total_round_actor_hidden_data = []

        self.current_round_data = []
        self.current_round_action = []
        self.current_round_action_dist_data = []
        self.current_round_actor_hidden_data = []
        logger.info('create new data helper')

    def put(self, data):
        if(len(data) == 1):
            logger.info('put data at game reset')
            if len(self.current_round_data) > 0 and len(self.current_round_data[-1]) == 1:
                logger.info('game reset data exists')
        self.current_round_data.append(data)

    def put_action(self, action):
        self.current_round_action.append(action)

    def put_action_dist(self, action_dist):
        self.current_round_action_dist_data.append(action_dist)

    def put_actor_hidden_data(self, hidden_data):
        self.current_round_actor_hidden_data.append(hidden_data)

    def finish_round(self):
        self.total_round_data.append(self.current_round_data)
        self.current_round_data = []

        self.total_round_action_data.append(self.current_round_action)
        self.current_round_action = []

        self.total_round_action_dist_data.append(self.current_round_action_dist_data)
        self.current_round_action_dist_data = []

        self.total_round_actor_hidden_data.append(self.current_round_actor_hidden_data)
        self.current_round_actor_hidden_data = []

    def reset(self):
        self.total_round_data = []
        self.current_round_data = []

        self.total_round_action_data = []
        self.current_round_action = []

        self.total_round_action_dist_data = []
        self.current_round_action_dist_data = []

        self.total_round_actor_hidden_data = []
        self.current_round_actor_hidden_data = []
