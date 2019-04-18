import wget

print('Downloading Skip-Thought Model ...........')
 
wget.download('http://www.cs.toronto.edu/~rkiros/models/dictionary.txt')
print('\ndictionary.txt Downloaded!')

wget.download('http://www.cs.toronto.edu/~rkiros/models/utable.npy')
print('\nutable.npy Downloaded!')

wget.download('http://www.cs.toronto.edu/~rkiros/models/btable.npy')
print('\nbtable.npy Downloaded!')

wget.download('http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz')
print('\nuni_skip.npz Downloaded!')

wget.download('http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl')
print('\nuni_skip.npz.pkl Downloaded!')

wget.download('http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz')
print('\nbi_skip.npz Downloaded!')

wget.download('http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl')
print('\nbi_skip.npz.pkl Downloaded!')

print('Download Completed ............')