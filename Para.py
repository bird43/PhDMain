from paramiko import client

class ssh:
    client = None
 
    def __init__(self, address, username, password):
        print("Connecting to server.")
        self.client = client.SSHClient()
        self.client.set_missing_host_key_policy(client.AutoAddPolicy())
        self.client.connect(address, username=username, password=password, look_for_keys=False)
 
    def sendCommand(self, command):
        if(self.client):
            stdin, stdout, stderr = self.client.exec_command(command)
            while not stdout.channel.exit_status_ready():
                # Print data when available
                if stdout.channel.recv_ready():
                    alldata = stdout.channel.recv(1024)
                    prevdata = b"1"
                    while prevdata:
                        prevdata = stdout.channel.recv(1024)
                        alldata += prevdata
 
                    #print(str(alldata, "utf8"))
        else:
            print("Connection not opened.")

def main():
    connection = ssh("192.168.56.101", "ben", "starscape")
    #connection.sendCommand("cd ~/Dropbox/Robotics/ABB/DriveSoftware && mkdir testssh")
    connection.sendCommand("cd ~/Dropbox/Robotics/ABB/DriveSoftware && xvfb-run meshlabserver -i TotalCloudKM090.xyz -o ssh2.off -s Del.mlx")
    print "done"

if __name__ == "__main__":
    main()