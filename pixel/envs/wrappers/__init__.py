"""Module of wrapper classes."""
from gym import error
from pixel.envs.wrappers.atari_preprocessing import AtariPreprocessing
from pixel.envs.wrappers.autoreset import AutoResetWrapper
from pixel.envs.wrappers.clip_action import ClipAction
from pixel.envs.wrappers.filter_observation import FilterObservation
from pixel.envs.wrappers.flatten_observation import FlattenObservation
from pixel.envs.wrappers.frame_stack import FrameStack, LazyFrames
from pixel.envs.wrappers.gray_scale_observation import GrayScaleObservation
from pixel.envs.wrappers.human_rendering import HumanRendering
from pixel.envs.wrappers.normalize import NormalizeObservation, NormalizeReward
from pixel.envs.wrappers.order_enforcing import OrderEnforcing
from pixel.envs.wrappers.record_episode_statistics import RecordEpisodeStatistics
from pixel.envs.wrappers.record_video import RecordVideo, capped_cubic_video_schedule
from pixel.envs.wrappers.render_collection import RenderCollection
from pixel.envs.wrappers.rescale_action import RescaleAction
from pixel.envs.wrappers.resize_observation import ResizeObservation
from pixel.envs.wrappers.step_api_compatibility import StepAPICompatibility
from pixel.envs.wrappers.time_aware_observation import TimeAwareObservation
from pixel.envs.wrappers.time_limit import TimeLimit
from pixel.envs.wrappers.transform_observation import TransformObservation
from pixel.envs.wrappers.transform_reward import TransformReward
from pixel.envs.wrappers.vector_list_info import VectorListInfo
