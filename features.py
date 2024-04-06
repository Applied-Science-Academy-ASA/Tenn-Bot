import cv2
import numpy as np

def new_features(masked, mask):
    knob_value = 0.4
    division_pairs = [[2, 1], [2, 0], [0, 1]]
    
    average_rgb = getAverageRgb(masked, mask)
    average_rgb_ratios = getAverageRgbRatios(average_rgb, division_pairs)
    
    #histograms = getHist(masked, mask)
    #histogram_ratios = getHistogramRatios(histograms, division_pairs)

    #histogram_intersection = getHistogramIntersection(histograms, division_pairs, knob_value, False)
    #histogram_intersection_average = getHistogramIntersectionAverage(histograms, division_pairs, knob_value, True)
    
    #average_intersection = getAverageIntersection(histograms, knob_value, False)
    
    #features = average_rgb + average_rgb_ratios + histogram_ratios + histogram_intersection + histogram_intersection_average + average_intersection
    features = average_rgb + average_rgb_ratios

    return features

def getAverage(matrix, beginning):
    num_rows = len(matrix)
    weighted_sum = 0
    for i in range(0,num_rows):
        weighted_sum += matrix[i][0] * ((i + beginning) / 256.0)
    return weighted_sum / np.sum(matrix)

def getHist(masked, mask):
    histograms = []
    
    for x in range(0,3):
        hist = cv2.calcHist([masked], [x], mask, [256],[0.0,255.0])
        hist = hist / np.sum(hist)
        histograms.append(hist)
    
    return histograms

def getAverageRgb(masked, mask):
    masked = masked / 255
    mask = mask / 255
    
    Bs, Gs, Rs = ([],[],[])
    for x in masked:
        for y in x:
            Bs.append(y[0])
            Gs.append(y[1])
            Rs.append(y[2])
    cv_sums = (sum(Bs), sum(Gs), sum(Rs))
    
    numValid = np.sum(mask)
    result = []
    for x in range(0,3):
        result.append(cv_sums[x]/numValid)
    
    return result

def getAverageRgbRatios(average_rgb, division_pairs):
    result = []

    for division_pair in division_pairs:
        result.append(average_rgb[division_pair[0]]/average_rgb[division_pair[1]])
    
    return result

def getHistogramRatios(histograms, division_pairs):
    result = []
    
    for division_pair in division_pairs:
        result.append(np.sum(cv2.absdiff(histograms[division_pair[0]],histograms[division_pair[1]])))
    
    return result

def getHistogramIntersection(histograms, division_pairs, knob_value, isBefore):
    result = []

    for division_pair in division_pairs:
        idx = int(knob_value * 256)
        num_rows = len(histograms[division_pair[0]])

        if isBefore:
            result.append(
                (np.sum(histograms[division_pair[0]][0:idx]) 
                /np.sum(histograms[division_pair[1]][0:idx]))
            )
        else :
            result.append(
                (np.sum(histograms[division_pair[0]][idx:num_rows])
                /np.sum(histograms[division_pair[1]][idx:num_rows]))
            )

    return result

def getHistogramIntersectionAverage(histograms, division_pairs, knob_value, isBefore):
    result = []

    for division_pair in division_pairs:
        idx = int(knob_value * 256)
        num_rows = len(histograms[division_pair[0]])

        if isBefore:
            result.append(
                (getAverage(histograms[division_pair[0]][0:idx],0) 
                /getAverage(histograms[division_pair[1]][0:idx],0))
            )
        else :
            result.append(
                (getAverage(histograms[division_pair[0]][idx:num_rows],idx)
                /getAverage(histograms[division_pair[1]][idx:num_rows],idx))
            )

    return result

def getAverageIntersection(histograms, knob_value, isBefore):
    result = []

    for i in range(0, 3):
        idx = int(knob_value * 256)
        num_rows = len(histograms[i])

        if (isBefore):
            result.push_back(getAverage(histograms[i].rowRange(0, idx), 0))
        else:
            result.append(getAverage(histograms[i][idx:num_rows], idx))
    
    return result
