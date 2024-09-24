import numpy as np
import torch
from pyftg.aiinterface.ai_interface import AIInterface
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.audio_data import AudioData
from pyftg.models.frame_data import FrameData
from pyftg.models.key import Key
from pyftg.models.round_result import RoundResult
from pyftg.models.screen_data import ScreenData

from src.config import GATHER_DEVICE
from src.dataclasses.collect_data_helper import CollectDataHelper
from src.models.recurrent_actor import RecurrentActor
from src.models.recurrent_critic import RecurrentCritic
from loguru import logger


class SoundAgent(AIInterface):
    def __init__(self, **kwargs):
        self.actor: RecurrentActor = kwargs.get('actor')
        self.critic: RecurrentCritic = kwargs.get('critic')
        self.device = GATHER_DEVICE
        self.collect_data_helper = kwargs.get('collect_data_helper')
        self.rnn = kwargs.get('rnn')
        self.trajectories_data = None

        self.actions = "AIR_A", "AIR_B", "AIR_D_DB_BA", "AIR_D_DB_BB", "AIR_D_DF_FA", "AIR_D_DF_FB", "AIR_DA", "AIR_DB", \
                       "AIR_F_D_DFA", "AIR_F_D_DFB", "AIR_FA", "AIR_FB", "AIR_UA", "AIR_UB", "BACK_JUMP", "BACK_STEP", \
                       "CROUCH_A", "CROUCH_B", "CROUCH_FA", "CROUCH_FB", "CROUCH_GUARD", "DASH", "FOR_JUMP", "FORWARD_WALK", \
                       "JUMP", "NEUTRAL", "STAND_A", "STAND_B", "STAND_D_DB_BA", "STAND_D_DB_BB", "STAND_D_DF_FA", \
                       "STAND_D_DF_FB", "STAND_D_DF_FC", "STAND_F_D_DFA", "STAND_F_D_DFB", "STAND_FA", "STAND_FB", \
                       "STAND_GUARD", "THROW_A", "THROW_B"
        self.audio_data = None
        self.raw_audio_memory = None
        self.just_inited = True
        self.pre_framedata: FrameData = None
        self.nonDelay: FrameData = None

        if self.rnn:
            self.actor.get_init_state(GATHER_DEVICE)
            self.critic.get_init_state(GATHER_DEVICE)
        self.round_count = 0
        self.n_frame = kwargs.get('n_frame')
    
    def name(self) -> str:
        return self.__class__.__name__
    
    def is_blind(self) -> bool:
        return False

    def initialize(self, gameData, player):
        # Initializng the command center, the simulator and some other things
        self.inputKey = Key()
        self.frameData = FrameData()
        self.cc = CommandCenter()
        self.player = player  # p1 == True, p2 == False
        self.gameData = gameData
        self.isGameJustStarted = True

    def close(self):
        pass

    def get_non_delay_frame_data(self, non_delay: FrameData):
        self.pre_framedata = self.nonDelay if self.nonDelay is not None else non_delay
        self.nonDelay = non_delay

    def get_information(self, frame_data: FrameData, is_control: bool):
        # Load the frame data every time getInformation gets called
        self.frameData = frame_data
        self.cc.set_frame_data(self.frameData, self.player)
        # nonDelay = self.frameData
        self.isControl = is_control
        self.currentFrameNum = self.frameData.current_frame_number  # first frame is 14

    def round_end(self, round_result: RoundResult):
        logger.info(round_result.remaining_hps[0])
        logger.info(round_result.remaining_hps[1])
        logger.info(round_result.elapsed_frame)
        self.just_inited = True
        obs = self.raw_audio_memory
        if obs is not None:
            self.collect_data_helper.put([obs, 0, True, None])
        # final step
        # state = torch.tensor(obs, dtype=torch.float32)
        terminal = 1
        true_reward = self.get_reward()
        self.collect_data_helper.finish_round()
        self.raw_audio_memory = None
        self.round_count += 1 
        logger.info('Finished {} round'.format(self.round_count))
    
    def game_end(self):
        pass

    def get_screen_data(self, screen_data: ScreenData):
        pass

    def input(self):
        return self.inputKey

    @torch.no_grad()
    def processing(self):
        # process audio
        try:
            np_array = np.frombuffer(self.audio_data.raw_data_bytes, dtype=np.float32)
            raw_audio = np_array.reshape((2, 1024))
            raw_audio = raw_audio.T
            raw_audio = raw_audio[:800, :]
        except Exception:
            raw_audio = np.zeros((800, 2))
        if self.raw_audio_memory is None:
            self.raw_audio_memory = raw_audio
        else:
            self.raw_audio_memory = np.vstack((raw_audio, self.raw_audio_memory))
            self.raw_audio_memory = self.raw_audio_memory[:800 * self.n_frame, :]

        # append so that audio memory has the first shape of n_frame
        increase = (800 * self.n_frame - self.raw_audio_memory.shape[0]) // 800
        for _ in range(increase):
            self.raw_audio_memory = np.vstack((np.zeros((800, 2)), self.raw_audio_memory))

        obs = self.raw_audio_memory
        if self.just_inited:
            self.just_inited = False
            if obs is None:
                obs = np.zeros((800 * self.n_frame, 2))
            self.collect_data_helper.put([obs])
            terminal = 1
        elif obs is None:
            obs = np.zeros((800 * self.n_frame, 2))
            self.collect_data_helper.put([obs])
        else:
            terminal = 0
            reward = self.get_reward()
            self.collect_data_helper.put([obs, reward, False, None])

        # get action
        state = torch.tensor(obs, dtype=torch.float32)
        action_dist = self.actor.forward(state.unsqueeze(0).to(self.device), terminal=torch.tensor(terminal).float())
        action = action_dist.sample()

        # put to helper
        self.collect_data_helper.put_action(action)
        if self.rnn:
            self.collect_data_helper.put_actor_hidden_data(self.actor.hidden_cell.squeeze(0).to(self.device))
        
        self.inputKey.empty()
        self.cc.skill_cancel()
        self.cc.command_call(self.actions[action])
        self.inputKey = self.cc.get_skill_key()

    def get_reward(self):
        offence_reward = self.pre_framedata.get_character(not self.player).hp - self.nonDelay.get_character(not self.player).hp
        defence_reward = self.nonDelay.get_character(self.player).hp - self.pre_framedata.get_character(self.player).hp
        return offence_reward + defence_reward

    def set_last_hp(self):
        self.last_my_hp = self.nonDelay.get_character(self.player).hp
        self.last_opp_hp = self.nonDelay.get_character(not self.player).hp

    def get_audio_data(self, audio_data: AudioData):
        self.audio_data = audio_data
    
    def reset(self):
        self.collect_data_helper = CollectDataHelper()
