# Setting up experiments in the cloud vm

## Pre-requisites
SSH to the cloud terminal. Linux installed on the vm - preferably debian.

1. run the following on first setup
```bash
sudo apt install tmux git python3.11-venv
git clone https://github.com/self1am/FL_CognitiveDefence.git
cd FL_CognitiveDefence
```

2. Once cloned, switch to the specified branch
```bash
git switch copilot/implement-adaptive-attacks
```

3. Now create the virtual environment
```bash
python3 -m venv fl_env
source fl_env/bin/activate
pip install -r requirements.txt
```

The previous step would have installed all the specified packages and libraries, and you're all set to run the experiments

## Running Experiments

1. use tmux to create a session
```bash
tmux new-session -d -s experiment -c .
```

2. attach the created session
```bash 
tmux attach -t experiment
```

3. now in the attahced session run specified experiments
> assuming you want to run the static attack with no defence scenario
```bash
python -m src.orchestration.simulation_runner --config experiments/configs/static_attacks_no_defence_STRONG.yaml
```

> Once you run this, you can close the terminal. You may connect to ssh again and attach the experiment to see the progress

```bash
tmux attach -t experiment
```

> Once the experiment has completed, just download the specific log file. For this scenario the log file would be at FL_CognitiveDefence/logs/static_attact_no_defence_STRONG.log
