from small_evo.smaller.train import Evaluator

weights_2 = [0.02562316, 0.37133697, 0.00982241, -0.02487976, -0.19998196, -0.03655382,
             -0.20460871, 0.06029608, -0.03905059, -0.09734652, -0.04051968, 0.11789757,
             0.10608996, 0.06954718, 0.04162831, 0.06654395, 0.15691736, -0.03615978]
weights_3 = [0.06603212, -0.08467746, 0.06253882, 0.11540868, -0.06345578, 0.13030864,
             -0.17183566, -0.18778044, -0.05617174, -0.06646609, -0.12728679, 0.0654319,
             0.15570651, 0.07487635, -0.01767378, 0.08531394, -0.08403758, -0.08138834]

weights_4_wahl = [-0.06935628, 0.06078843, -0.66082906, -0.50534815, 0.42725888, 0.40346524,
                  -1.07318775, 0.19004175, -0.04961645, 0.08676088, -0.10972614, 0.71822668,
                  0.2710733, -0.43031814, -0.31273709, 0.76265546, -0.52503067, 0.1228674]

if __name__ == '__main__':
    evaler = Evaluator(render=2, max_episode_steps=3000, deterministic=False)
    for episode_i in range(5):
        print(evaler.evaluate(weights_4_wahl))

# some result i guess 200 reward
# [ 0.09923301 -0.20626588  0.19587401 -0.14792714 -0.18294119  0.1533143
#  -0.00447617 -0.26682253  0.0278062  -0.20441508  0.18823991  0.03357062
#   0.23752833 -0.21154231 -0.00536211  0.01096144 -0.08628416  0.12551947]

# should be 600
# [ 0.02562316  0.37133697  0.00982241 -0.02487976 -0.19998196 -0.03655382
#  -0.20460871  0.06029608 -0.03905059 -0.09734652 -0.04051968  0.11789757
#   0.10608996  0.06954718  0.04162831  0.06654395  0.15691736 -0.03615978]
