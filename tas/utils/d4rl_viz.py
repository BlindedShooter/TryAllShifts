import numpy as np
import imageio


def visualize_traj(env, states):
    # Reset the environment to get the initial state
    state = env.reset()

    # Render the initial state
    frames = []

    # Iterate through the sequence of states
    for state in states:
        # Set the state of the environment
        qpos = state[:env.model.nq]
        qvel = state[env.model.nv:]

        env.set_state(qpos, qvel)

        # Render the state
        frame = env.render(mode='rgb_array')
        frames.append(frame)

    return frames


def images_to_video(imgs: list[np.ndarray], video_path: str, fps: int=24, **kwargs):
    writer = imageio.get_writer(video_path, fps=24, **kwargs)

    for img in imgs:
        writer.append_data(img)
    
    writer.close()