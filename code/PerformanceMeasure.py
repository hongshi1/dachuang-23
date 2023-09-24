# coding=utf-8
import numpy as np


class PerformanceMeasure():

    def __init__(self, real_list=None, pred_list=None, loc=None,  percentage=0.2, ranking="density", cost="loc"):
        self.real = real_list
        self.pred = pred_list
        self.loc = loc
        self.percentage = percentage
        self.ranking = ranking
        self.cost=cost

    def Performance(self):
        if (len(self.pred) != len(self.real)) or (len(self.pred) != len(self.loc) or (len(self.loc) != len(self.real))):

            exit()

        M = len(self.real)
        L = sum(self.loc)
        P = sum([1 if i > 0 else 0 for i in self.real])
        m=None
        Q = sum(self.real)



        if (self.ranking == "density" and self.cost=='loc'):



            density = []
            for i in range(len(self.pred)):
                if self.loc[i] != 0:
                    density.append(self.pred[i] / self.loc[i])
                else:
                    density.append(-100000000)

            sort_axis = np.argsort(density)[::-1]

            sorted_pred = np.array(self.pred)[sort_axis]
            sorted_real = np.array(self.real)[sort_axis]
            sorted_loc = np.array(self.loc)[sort_axis]



            locOfPercentage = self.percentage * L
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    m = i
                    break
                elif (sum_ == locOfPercentage):
                    m = i + 1
                    break


            PMI=m/M


        tp = sum([1 if sorted_real[j] > 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
        fn = sum([1 if sorted_real[j] > 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
        fp = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
        tn = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
        # print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))
        if (tp + fn + fp + tn == 0):
            Precisionx = 0
        else:
            Precisionx = (tp + fn) / (tp + fn + fp + tn)

        if (P == 0):
            Recallx = 0
        else:
            Recallx = (tp + fn) / P

        if (P == 0 and PMI == 0):
            recallPmi = 0
        else:
            recallPmi = Recallx / PMI

        if (Recallx + Precisionx==0):
            F1x = 0
        else:
            F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)


        IFLA = 0
        IFMA = 0

        for i in range(m):
            if (sorted_real[i] > 0):
                break
            else:
                IFLA += sorted_loc[i]

                IFMA += 1

        PofB = sum([sorted_real[j] if sorted_real[j] > 0 else 0 for j in range(m)]) / Q

        if (Q == 0 and PMI == 0):
            PofBPmi = 0
        else:
            PofBPmi = PofB / PMI



        return   Precisionx, Recallx,F1x, IFMA,PMI, recallPmi,PofB,PofBPmi


    def getPofb(self):
        if (len(self.pred) != len(self.real)) or (len(self.pred) != len(self.loc) or (len(self.loc) != len(self.real))):

            exit()

        M = len(self.real)
        L = sum(self.loc)
        P = sum([1 if i > 0 else 0 for i in self.real])
        m=None
        Q = sum(self.real)



        if (self.ranking == "density" and self.cost=='loc'):



            density = []
            for i in range(len(self.pred)):
                if self.loc[i] != 0:
                    density.append(self.pred[i] / self.loc[i])
                else:
                    density.append(-100000000)

            sort_axis = np.argsort(density)[::-1]

            sorted_pred = np.array(self.pred)[sort_axis]
            sorted_real = np.array(self.real)[sort_axis]
            sorted_loc = np.array(self.loc)[sort_axis]



            locOfPercentage = self.percentage * L
            sum_ = 0
            for i in range(len(sorted_loc)):
                sum_ += sorted_loc[i]
                if (sum_ > locOfPercentage):
                    m = i
                    break
                elif (sum_ == locOfPercentage):
                    m = i + 1
                    break


            PMI=m/M


        tp = sum([1 if sorted_real[j] > 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
        fn = sum([1 if sorted_real[j] > 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
        fp = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] > 0 else 0 for j in range(m)])
        tn = sum([1 if sorted_real[j] <= 0 and sorted_pred[j] <= 0 else 0 for j in range(m)])
        # print('tp:{0},fn:{1},fp:{2},tn:{3}'.format(tp,fn,fp,tn))
        if (tp + fn + fp + tn == 0):
            Precisionx = 0
        else:
            Precisionx = (tp + fn) / (tp + fn + fp + tn)

        if (P == 0):
            Recallx = 0
        else:
            Recallx = (tp + fn) / P

        if (P == 0 and PMI == 0):
            recallPmi = 0
        else:
            recallPmi = Recallx / PMI

        if (Recallx + Precisionx==0):
            F1x = 0
        else:
            F1x = 2 * Recallx * Precisionx / (Recallx + Precisionx)


        IFLA = 0
        IFMA = 0

        for i in range(m):
            if (sorted_real[i] > 0):
                break
            else:
                IFLA += sorted_loc[i]

                IFMA += 1

        PofB = sum([sorted_real[j] if sorted_real[j] > 0 else 0 for j in range(m)]) / Q

        if (Q == 0 and PMI == 0):
            PofBPmi = 0
        else:
            PofBPmi = PofB / PMI



        return   PofB

    def POPT(self):


        Q = sum(self.real)

        if (self.ranking == "density" and self.cost == 'loc'):

            pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(self.loc, self.pred)]
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost=self.loc
            xcostsum=sum(xcost)


        elif (self.ranking == "defect" and self.cost == 'module'):

            pred_index = self.pred
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            ###这里不会
            xcost = [1 for i in range(len(self.pred))]
            xcostsum = sum(xcost)


        elif (self.ranking == "complexitydensity" and self.cost == 'cc'):

            pred_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(self.cc, self.pred)]
            pred_index = list(np.argsort(pred_index))
            pred_index.reverse()
            xcost = self.cc
            xcostsum = sum(xcost)

        else:

            exit()

        optimal_index = [j / i if j != 0 and i != 0 else 0 for i, j in zip(xcost, self.real)]
        optimal_index = list(np.argsort(optimal_index))
        optimal_index.reverse()

        optimal_X = [0]
        optimal_Y = [0]
        for i in optimal_index:
            optimal_X.append(xcost[i] / xcostsum + optimal_X[-1])
            optimal_Y.append(self.real[i] / Q + optimal_Y[-1])

        wholeoptimal_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(optimal_X, optimal_Y):
            if x != prev_x:
                wholeoptimal_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('wholeoptimalx', (x - prev_x))
                #print('wholeoptimaly', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('wholeoptimalauc', wholeoptimal_auc)

        pred_X = [0]
        pred_Y = [0]
        for i in pred_index:
            pred_X.append(xcost[i]/ xcostsum + pred_X[-1])
            pred_Y.append(self.real[i] / Q + pred_Y[-1])

        wholepred_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(pred_X, pred_Y):
            if x != prev_x:
                wholepred_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('predx', (x - prev_x))
                #print('predy', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('wholepredauc',wholepred_auc)

        optimal_index.reverse()
        mini_X = [0]
        mini_Y = [0]
        for i in optimal_index:
            mini_X.append(xcost[i]/ xcostsum + mini_X[-1])
            mini_Y.append(self.real[i] / Q + mini_Y[-1])

        wholemini_auc = 0.
        prev_x = 0
        prev_y = 0
        for x, y in zip(mini_X, mini_Y):
            if x != prev_x:
                wholemini_auc += (x - prev_x) * (y + prev_y) / 2.
                #print('wholeworstpredx', (x - prev_x))
                #print('wholeworstpredy', (y + prev_y))
                prev_x = x
                prev_y = y

        #print('worstwholeauc',wholemini_auc)

        wholemini_auc = 1 - (wholeoptimal_auc - wholemini_auc)
        wholenormOPT = ((1 - (wholeoptimal_auc - wholepred_auc)) - wholemini_auc) / (1 - wholemini_auc)

        return wholenormOPT


if __name__ == '__main__':
    real = [0, 4, 1, 0, 1, 0, 1, 0, 0, 0]
    pred = [60, 20, 22.5, 9, 9.9, 0, -80, -270, -100, -150]
    loc = [200, 100, 150, 90, 110, 500, 400, 900, 250, 300]
    popt = PerformanceMeasure(real, pred, loc, 0.2,'density','loc').POPT()
    print(popt)

