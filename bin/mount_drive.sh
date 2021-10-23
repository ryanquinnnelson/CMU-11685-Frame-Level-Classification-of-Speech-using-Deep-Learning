#!/usr/bin/env bash

# Author: ryanquinnnelson

# Follows the steps outlined on StackOverflow:
# https://stackoverflow.com/questions/45167717/mounting-a-nvme-disk-on-aws-ec2/64709212#64709212

echo "Listing block devices..."
lsblk
echo

# the output should be /dev/nvme1n1: data for empty drives
sudo file -s /dev/nvme1n1


if [[ $(sudo file -s /dev/nvme1n1 | grep -L ": data") ]]; then
	echo "Drive is not empty."
else
	echo "Drive is empty. Mounting drive..."

	# format
	sudo mkfs -t xfs /dev/nvme1n1

	# create a folder in current directory and mount the nvme drive
	sudo mkdir /data
	sudo mount /dev/nvme1n1 /data
	echo

	# change owner and group
	sudo chown ubuntu /data
	sudo chgrp ubuntu /data
fi


# check existence
echo "Checking amount of free disk space available..."
df -h

