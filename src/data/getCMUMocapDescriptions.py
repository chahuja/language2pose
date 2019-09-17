from subprocess import Popen, PIPE
import os
from pathlib import Path

link = "http://mocap.cs.cmu.edu/search.php?subjectnumber={}&motion=%%%&maincat=%&subcat=%&subtext=yes"
for i in range(1, 145):
  if i != 4: ## subject 4 does not exist
    command = ['wget', link.format(i), '-O', "../../dataset/cmu-pose/all_asfamc/descriptions/{:02d}.csv".format(i)]
    process = Popen(command, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    print(i)
