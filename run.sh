# mddqn
python main.py --attacker static --decider mddqn --num_episodes 2000 --max_episode_step 50 --title test

python main.py --attacker yoyo --decider mddqn --num_episodes 2000 --max_episode_step 50 --title test

python main.py --attacker random --decider mddqn --num_episodes 2000 --max_episode_step 50 --title test

# dqn
python main.py --attacker static --decider dqn --num_episodes 2000 --max_episode_step 50 --title test

python main.py --attacker yoyo --decider dqn --num_episodes 2000 --max_episode_step 50 --title test

python main.py --attacker random --decider dqn --num_episodes 2000 --max_episode_step 50 --title test

# hpa
python main.py --attacker static --decider hpa --num_episodes 100 --max_episode_step 50 --title test

python main.py --attacker yoyo --decider hpa --num_episodes 100 --max_episode_step 50 --title test

python main.py --attacker random --decider hpa --num_episodes 100 --max_episode_step 50 --title test

# greedy
python main.py --attacker static --decider greedy --num_episodes 100 --max_episode_step 50 --title test

python main.py --attacker yoyo --decider greedy --num_episodes 100 --max_episode_step 50 --title test

python main.py --attacker random --decider greedy --num_episodes 100 --max_episode_step 50 --title test

# log console
python main.py --attacker static --decider mddqn --num_episodes 2000 --max_episode_step 50 --title test --log_console true > ./txt/demo.log 2>&1

# draw
python draw.py --jsonname v1-hpa-random-0729-105900-v1-hpa-random-0729-105900

python draw.py --jsonname v1-hpa-static-0729-105844-v1-hpa-static-0729-105844

python draw.py --jsonname v1-hpa-yoyo-0729-105853-v1-hpa-yoyo-0729-105853