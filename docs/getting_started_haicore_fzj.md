## Getting started at with Haicore@FZJ

1. Sign up at [JuDoor](https://judoor.fz-juelich.de/register).
1. Join the compute project `hai_countmein` following this [link](https://judoor.fz-juelich.de/projects/join/hai_countmein). You will be admitted to the project in a short while 
1. Get access by SSH. To do that, please follow the steps indicated in the [Section Setting up SSH](#setting-up-ssh). Now you should be able to ssh to the machines and start compute jobs via Slurm.
1. Your home directory is very limited in space. We thus create a personal directory in the project directory and link it to the home directory. 
```bash
mkdir -p /p/project/hai_countmein/$USER
ln -s /p/project/hai_countmein/$USER ~/hai_countmein
```
Now you can use the linked directory `~/hai_countmein` for all activities regarding the challenge.

5. Visit the Jupyter-JSC page and log into Jupyter: http://jupyter-jsc.fz-juelich.de/
6. You will find the data in the directory `/p/project/hai_countmein/data`

## Setting up SSH
##### SSH Client
Before you can actually start, it is required that an SSH client is installed on your machine. On both, Mac and Linux, an SSH client should be installed by default. On Windows, it is recommended to install the Windows Subsystem for Linux (WSL). On older Windows versions without WSL, you have to install a terminal emulator like PuTTY.
##### Generate Key Pair
As a first step, you will create a key pair for public/private key authentification. Then, you will register the public keys for access to Juwels using the JuDoor web page. To do so, it is required to add a meaningful restriction of the range of IPs or hostnames that are allowed to connect to Juwels. Finally, you will be able to connect to Juwels. This exercise guides you through the process that is explained in more detail in the [Juwels access documentation pages](https://apps.fz-juelich.de/jsc/hps/juwels/access.html).

Execute the following command in the command line to create an ED25519 key pair directly into your .ssh directory.
```bash
ssh-keygen -a 100 -t ed25519 -f ~/.ssh/id_ed25519
```
On Windows, you must define a different storage location for the key pair, but otherwise the command works. In WSL you can execute the command right away.

The command generated two keys, a public one and a private one. The public key (ending in .pub) is similar to your hand-written signature: you may give it to others who can then use it to confirm your identity. The private key (we called it id_ed25519) must not be shared. Continuing with the hand-written signature analogy, the private key is the way you write your signature. Just as you would not give others the ability to perfectly copy your hand-written signature, you should under no circumstance publicize your private key.

##### The From-Clause

Before you can add your public SSH key to the list of authorized SSH keys for Juwels, you must create a valid from-clause that meaningfully restricts the range of IPs. You have several options to that, e.g. check the IP range of your internet service provider (ISP). If you know the IP of your ISP, or if you can connect to a VPN giving you a fixed IP range (FZ Jülich's VPN is an example, but other institutions work as well), this is very easy. You can directly use the IP range as a from-clause. For FZ Jülich your from-clause would be:
```
from="134.94.0.0/16"
```
Note that the /16 indicates the subnet, hence all adresses of the form `134.94.\*.\*` will be allowed. If you use this option, you can directly jump to the point Register your public key.

We also show here the slightly more difficult steps to create a from-clause based on reverse DNS lookup.

1.  Visit the JuDoor page. Earlier, you should have visited this page to register and get access to the compute resources. Under the header Systems, find **juwels -> Manage SSH-keys** and navigate to it. On this page, your IP should be visible. Example: Your current IP address is 37.201.214.241.
2. Perform a reverse DNS search of your IP and extract the DNS name (the field Name) associated with your IP. Type into your command line:
```bash
nslookup <your-ip>
```
Example results:
```
Name:    aftr-37-201-214-241.unity-media.net 
```
Guess a wildcard pattern that will likely apply for all future connections. For example `*.unity-media.net`.

##### Register your Public Key

Now, you can register your key pair in JuDoor: Create a from-clause from your wildcard expression and enter it into the field Your public key and options string, but do not confirm yet. Then, open your public key file `~/.ssh/id_ed25519.pub` and copy your public key into the same field (making sure there is a single space between the from-clause and the contents of the file) and select Start upload of SSH-Keys. Note the file ending `.pub`!

Example line:
```
from="*.unity-media.net" ssh-ed25519 AAAAasdbmnsowrmnsdigninmnmnasdta username@HOSTNAME
```
After a few minutes, your newly added SSH key should be available. Note that JuDoor writes the file `~/.ssh/authorized_keys` in your Juwels home directory, thus manually added SSH keys will automatically be overwritten.

Finally, you can log into Juwels, using 
```bash
ssh -i <path>/<to>/id_ed25519 username@juwels-booster.fz-juelich.de
```
If you have created the key pair in `~/.ssh/` it is possible to omit the `-i` option as ssh will try all keys in your `.ssh` directory by default. Your username is identical to the username in the JuDoor website, typically lastname1.

