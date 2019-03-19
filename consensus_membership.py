import numpy as np
import string
print("enter the file names")
files=raw_input()
file_name=files.split(" ")
no_files=len(file_name)
result=[]
temp=[]
base="/home/debomit/Desktop/hh/"
max=0
for i in range(no_files):
	with open(base+file_name[i],"r") as fin:
		mem=fin.readlines()
		line=[]
		for l in range(1,len(mem)):
			lines=str(mem[l].replace("\n",''))
			lines=str(lines.replace('.000000',''))
			#print(lines)
			temp=(lines.split("  "))
			#temp=temp[1:-1]
			#print(temp)
			temp=list(map(int,temp))
			#print(temp)			
			line.append(temp)
			if(temp[1]>=max):
				max=temp[1]
		result.append(line)
#print(len(result))
new_file=["membership_consensus1.txt","membership_consensus2.txt","membership_consensus3.txt","membership_consensus4.txt","membership_consensus5.txt","membership_consensus6.txt"]
for i in range(no_files):
	consensus=[[0]*len(result[0]) for l in range(max)]
	for line in result[i]:
		consensus[line[1]-1][line[0]]=1
	np.savetxt(new_file[i], consensus, delimiter='	',  newline='\n',fmt='%0.4f')
		
