feature = []
for line in open('featmap.txt'):
	feature.append([line.split('\t')[0],line.split('\t')[1],0])
for line in open('dump.nice.txt'):
	for f in range(len(feature)):
		if feature[f][1] in line: feature[f][2] += 1
for f in feature:
	print f[0],f[1],f[2]