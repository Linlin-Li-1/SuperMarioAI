```diff
- AS OF APRIL 11TH, 2020, I WILL STOP UPDATING ON THIS REPOSITORY 
- UNTIL I GET FURTHER RESPONSE FROM USERS OR COLLABORATORS. 
- TUNING AND TRAINING CAN BE DRAMASTICALLY TIME-CONSUMING ON ONE SINGLE CPU,
- ESPECIALLY WITH THE NEED OF FIXING CODES SIMULTANEOUSLY.
```

# SuperMarioAI
An AI which plays Super Mario


## About Git

#### To clone the repository
```shell
git clone https://github.com/LauBok/SuperMarioAI.git
```

#### To commit a change
You should first make sure your current directory is the repository directory.
```shell
git add file # for files not tracked
git commit -m "message"
```

#### To push or pull
```shell
git pull
git push # Make sure to commit before push
```

#### To set up a new branch
```shell
git branch branch_name
```

#### To move to some branch
```shell
git checkout branch_name
```

If you want to set up a new branch and move to that branch, you can write together as
```shell
git checkout -b branch_name
```

#### To merge a branch to master
```shell
git checkout master
git merge branch_name
```

#### To delete a branch
If you have successfully merged from a branch, you may not need it anymore.
You can delete the branch via this instruction.
```shell
git branch -d branch_name
```

#### To resolve a conflict
If there is a conflict in the merge process, you can check the status.
```shell
git status
```
Then, you can resolve the conflict and use `git add` to mark it as resolved.
You can also use `git mergetool` to open a GUI for resolving merge conflicts.


## Install the environment
The preferred installation is through `pip`:
```shell
pip install gym-super-mario-bros
```

Then you can check whether it has been successfully installed by running this line in bash:
```shell
gym_super_mario_bros -e 'SuperMarioBros-v0' -m 'random'
```

## Install Tensorflow
You can install tensorflow through `pip` if your `pip` version is $>19.0$:
```shell
pip install tensorflow
```

Otherwise, you may want to first upgrade your `pip`.
```shell
pip install --upgrade pip
```

## References
- The official website of package `gym-super-mario-bros`: https://pypi.org/project/gym-super-mario-bros/.