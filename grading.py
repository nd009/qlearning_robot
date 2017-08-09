import mazeqlearning as mq


def test_world(filename, median_reward):
    student_reward = mq.maze_qlearning(filename)
    if student_reward < 1.5*median_reward:
        return "Reward too low, expected %s, found %s"%(median_reward,student_reward)
    else:
        return "pass"

print(test_world('testworlds/world01.csv', -29))
print(test_world('testworlds/world02.csv', -19))
print(test_world('testworlds/world03.csv', -80))
print(test_world('testworlds/world04.csv', -33))
print(test_world('testworlds/world05.csv', -24))
print(test_world('testworlds/world06.csv', -23))
print(test_world('testworlds/world07.csv', -26))
print(test_world('testworlds/world08.csv', -19))
print(test_world('testworlds/world09.csv', -20))
print(test_world('testworlds/world10.csv', -42))
