import skipthoughts
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

from collections import defaultdict
import pickle

"""
Step 1
file_caption = defaultdict(list)

with open('caption.txt') as fp:
	line = fp.readline()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	#line = [x.strip() for x in line]
	while(line):
		line = line.strip()
		print(line)
		file_name, caption =  line.split(',')
		file_caption[file_name].append(caption)
		line = fp.readline()
       

print(len(file_caption))
print(file_caption['1.png'])
print(file_caption['3.png'])

file_Name = "file_caption_map.pickle"
# open the file for writing
fileObject = open(file_Name,'wb') 

pickle.dump(file_caption,fileObject)   

# here we close the fileObject
fileObject.close()
"""

file_Name = "file_caption_map.pickle"
# we open the file for reading
fileObject = open(file_Name,'rb')  
# load the object from the file into var b
file_caption = pickle.load(fileObject)  

file_emedding = defaultdict(list)
for key, value in file_caption.iteritems():
	file_emedding[key] = encoder.encode(value, verbose=False)
	"""
	print(key, value)
	print(len(file_emedding[key]))
	print(file_emedding[key])
	print('-----------------------')
	"""
print(len(file_emedding))
print(file_emedding['1.png'])
print(file_emedding['3.png'])

file_Name = "file_caption_embedding.pickle"
# open the file for writing
fileObject = open(file_Name,'wb') 

pickle.dump(file_emedding,fileObject)   

# here we close the fileObject
fileObject.close()

	
	
