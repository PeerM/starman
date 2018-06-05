import retro

movie = retro.Movie('movies/6_1-1_with_shortcut.fm2')
movie.step()

env = retro.make("SuperMarioBros-Nes", None, use_restricted_actions=retro.Actions.ALL)
env.initial_state = movie.get_state()
env.reset()

frame_counter = 0
while movie.step():
    keys = []
    for i in range(env.num_buttons):
        keys.append(movie.get_key(i, 0))
    _obs, _rew, _done, _info = env.step(keys)
    if frame_counter % 10 == 0:
        env.render()
    frame_counter += 1
