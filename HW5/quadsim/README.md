# quadsim

Simulator for Dr. Beard's course EC EN 682R: Vision Enabled Estimation and Control of Multirotor Aircraft.

For instructions on setting up a ROS workspace with quadsim, head over to the [quadsim_ws](https://magiccvs.byu.edu/gitlab/quadrotorbook/quadsim_ws) repository. 
As explained there, you will want to make a personal fork of this repository and you should clone your fork into your local ROS workspace.


# Maintaining your fork of quadsim
After a successful setup of your local ROS workspace containing your fork of `quadsim`, the next step is to keep your fork of `quadsim` up to date with changes made to the base `quadsim` repo.
The following section explains a good workflow for doing just that.

## First Time Only - Adding an `upstream` Remote
Add the base repository as a secondary remote -- a common name given is `upstream`

```shell
git remote add upstream git@magiccvs.byu.edu:quadrotorbook/quadsim.git
```

## Every Time Thereafter
Fetch the remote (which we named `upstream`)

```shell
git checkout master
git fetch upstream 
# pull its changes into your fork's local master branch 
git pull upstream master
```

Now the base repository (`upstream`) has been merged in to your local branch of your fork. Fix any merge conflicts, then push the changes to your own remote (`origin`, i.e. the remote that is your personal fork on GitLab)

```shell
git push origin master
```

Now create a new branch where you will work on the next assignment. For example,

```shell
git checkout -b generate-trajectory
```

If there are changes to quadsim, then merge those changes to your fork's local master first:

```shell
git pull upstream master
```

Then you can merge your local master branch that was just updated into your "feature" branch:

```shell
git checkout generate-trajectory
git merge master
# Fix any merge conflicts
git commit -m "Merged upstream changes"
```

Once your sim works well enough for your liking, then you can merge that into your local master

```shell
git checkout master
git merge --squash generate-trajectory  
git commit
```

**Why the squash?** 

`--squash` will take your 57 commits from your feature branch with all those little typo fixes you made and condense them into one single, clean commit.

This workflow keeps your master branch looking nice and pretty. 
