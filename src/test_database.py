import pickle

num_user = pickle.load(open("num_user.p", "rb" ))
print(num_user)
train, testRatings, testNegatives = pickle.load(open(prepath + "mat.p", "rb" )), pickle.load(open(prepath + "testRatings.p")), pickle.load(open(prepath + "testNegatives.p"))
