def deabs(targets, outputs):
    chaabs= abs(outputs - targets)
    #print(chapow)
    return chaabs

def cul_mae(filename):
    with open(filename) as f:
        file=f.readlines()
        #print(file)
        true_value=[]
        pre_value=[]
        list_abs=[]
        for i in file:
            #print(i)
            float_i = float(i.split(' ')[0])
            true_value.append(float_i)  # 测试集的实际值
            float_j=float(i.split(' ')[1])
            pre_value.append(float_j)
            chaabs=deabs(float_i,float_j)
            list_abs.append(chaabs)
        # print(list_abs)
        # print(sum(list_abs)/len(list_abs))    #1.6119719999999989
        f.close()
        return sum(list_abs)/len(list_abs)
train_mae=cul_mae("train.dat")
print("Train set mae:",train_mae)
pre_mae=cul_mae("prediction.dat")
print("Test set mae:",pre_mae)
#
# with open('MAE_train.txt','a') as ff:
#     ff.write(str(sum(list_abs)/len(list_abs))+"\n")
#Train set
# 1.4678995000000006
# 1.3548555000000004
# 1.3100409999999993
# 1.9468469999999996
# 1.6640049999999997
# 1.4067834999999989
# 2.0946369999999996
# 1.8257889999999992
# 2.474537500000002
# 2.846421499999999



#Prediction set
# 2.4646182539682537
# 2.2269468253968263
# 2.254291269841271
# 3.026394444444445
# 2.4552119047619043
# 2.431796825396824
# 2.57921746031746
# 2.6446722222222214
# 2.8276500000000007
# 2.6467492063492073

# Train set mae: 1.5810280000000008
# Test set mae: 4.088809294871796