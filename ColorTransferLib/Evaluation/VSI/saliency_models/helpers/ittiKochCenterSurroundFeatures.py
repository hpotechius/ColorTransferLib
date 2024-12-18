import cv2
def compute(feature_pyramids):
    center_levels = [2, 3, 4]
    delta = [2, 3]
    Ccs_array = {
        0:[], 1:[]
    }
    Ics_array =[]
    Ocs_array = []
    for c in center_levels:
        for d in delta:
            s = c+d
            for i in range(0,2): # For calculating RG and BY channels
                Cc = feature_pyramids[c][i]
                Cs = -feature_pyramids[s][i]
                # to allow for chromatic opponency

                Cs_scaled = cv2.resize(Cs, (Cc.shape[1], Cc.shape[0]), interpolation=cv2.INTER_CUBIC)
                Ccs = (Cc - Cs_scaled) ** 2
                Ccs_array[i].append(Ccs)

            Ic = feature_pyramids[c][2]
            Is = feature_pyramids[s][2]

            Is_scaled = cv2.resize(Is, (Ic.shape[1], Ic.shape[0]), interpolation=cv2.INTER_CUBIC)
            Ics = (Ic - Is_scaled)**2
            Ics_array.append(Ics)

            for idx in range(0, len(feature_pyramids[c][3])):
                Oc = feature_pyramids[c][3][idx]
                Os = feature_pyramids[s][3][idx]

                Os_scaled = cv2.resize(Os, (Oc.shape[1], Oc.shape[0]), interpolation=cv2.INTER_CUBIC)
                Ocs = (Oc - Os_scaled) ** 2
                Ocs_array.append(Ocs)

    final_Feature_Array = {
        0:Ccs_array[0],
        1:Ccs_array[1],
        2:Ics_array,
        3:Ocs_array
    }
    return final_Feature_Array





