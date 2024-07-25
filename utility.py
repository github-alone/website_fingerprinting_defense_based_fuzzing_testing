import pickle as pickle
import numpy as np

def LoadGoodSampleCW():
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = './Dataset/good_samples/ClosedWorld/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load good sample
    with open(dataset_dir + '95w_20tra.pkl', 'rb') as handle:
        X_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + '95w_20lab.pkl', 'rb') as handle:
        y_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Testing data's shape : ", X_goodSample.shape)
    print ("y: Testing data's shape : ", y_goodSample.shape)

    # return X_train, y_train, X_valid, y_valid, X_test, y_test
    return X_goodSample, y_goodSample

def LoadGoodSampleOW():
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = './Dataset/good_samples/OpenWorld/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load good sample
    with open(dataset_dir + 'OpenWorld_95w_20tra.pkl', 'rb') as handle:
        X_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'OpenWorld_95w_20lab.pkl', 'rb') as handle:
        y_goodSample = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Testing data's shape : ", X_goodSample.shape)
    print ("y: Testing data's shape : ", y_goodSample.shape)

    # return X_train, y_train, X_valid, y_valid, X_test, y_test
    return X_goodSample, y_goodSample

# Load data for non-defended Dataset for CW setting
def LoadDataNoDefCW():
    print("Loading non-defended Dataset for closed-world scenario" )
    # Point to the directory storing data
    dataset_dir = 'Dataset/ClosedWorld/Nodef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load testing data
    with open(dataset_dir + 'X_test_NoDef.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_test_NoDef.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
    # return X_train, y_train,X_test, y_test

# Load data for non-defended Dataset for CW setting
def LoadDataWTFPADCW():

    print ("Loading WTF-PAD Dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../Dataset/ClosedWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load testing data
    with open(dataset_dir + 'X_test_WTFPAD.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_test_WTFPAD.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended Dataset for CW setting
def LoadDataWalkieTalkieCW():

    print ("Loading Walkie-Talkie Dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = '../Dataset/ClosedWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load testing data
    with open(dataset_dir + 'X_test_WalkieTalkie.pkl', 'rb') as handle:
        X_test = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_test_WalkieTalkie.pkl', 'rb') as handle:
        y_test = np.array(pickle.load(handle,encoding='iso-8859-1'))

    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)
    print ("X: Testing data's shape : ", X_test.shape)
    print ("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# Load data for non-defended Dataset for OW training
def LoadDataNoDefOW_Training():

    print ("Loading non-defended Dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = './Dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_NoDef.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_NoDef.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_NoDef.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_NoDef.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))


    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for non-defended Dataset for OW evaluation
def LoadDataNoDefOW_Evaluation():

    print ("Loading non-defended Dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = './Dataset/OpenWorld/NoDef/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load evaluating data
    with open(dataset_dir + 'X_test_Mon_NoDef.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'y_test_Mon_NoDef.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'X_test_Unmon_NoDef.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'y_test_Unmon_NoDef.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle,encoding='iso-8859-1')

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    print ("Data dimensions:")
    print ("X: Testing data_Mon's shape : ", X_test_Mon.shape)
    print ("y: Testing data_Mon's shape : ", y_test_Mon.shape)
    print ("X: Testing data_Unmon's shape : ", X_test_Unmon.shape)
    print ("y: Testing data_Unmon's shape : ", y_test_Unmon.shape)


    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# Load data for WTF-PAD Dataset for OW training
def LoadDataWTFPADOW_Training():

    print ("Loading WTF-PAD Dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../Dataset/OpenWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WTFPAD.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_WTFPAD.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_WTFPAD.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_WTFPAD.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))


    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for WTF-PAD Dataset for OW evaluation
def LoadDataWTFPADOW_Evaluation():

    print ("Loading WTF-PAD Dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../Dataset/OpenWorld/WTFPAD/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_WTFPAD.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'y_test_Mon_WTFPAD.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'X_test_Unmon_WTFPAD.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'y_test_Unmon_WTFPAD.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle,encoding='iso-8859-1')

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon

# Load data for WalkieTalkie Dataset for OW training
def LoadDataWalkieTalkieOW_Training():

    print ("Loading Walkie-Talkie Dataset for open-world scenario for training")
    # Point to the directory storing data
    dataset_dir = '../Dataset/OpenWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_train_WalkieTalkie.pkl', 'rb') as handle:
        X_train = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_train_WalkieTalkie.pkl', 'rb') as handle:
        y_train = np.array(pickle.load(handle,encoding='iso-8859-1'))

    # Load validation data
    with open(dataset_dir + 'X_valid_WalkieTalkie.pkl', 'rb') as handle:
        X_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))
    with open(dataset_dir + 'y_valid_WalkieTalkie.pkl', 'rb') as handle:
        y_valid = np.array(pickle.load(handle,encoding='iso-8859-1'))


    print ("Data dimensions:")
    print ("X: Training data's shape : ", X_train.shape)
    print ("y: Training data's shape : ", y_train.shape)
    print ("X: Validation data's shape : ", X_valid.shape)
    print ("y: Validation data's shape : ", y_valid.shape)

    return X_train, y_train, X_valid, y_valid

# Load data for WTF-PAD Dataset for OW evaluation
def LoadDataWalkieTalkieOW_Evaluation():

    print ("Loading Walkie-Talkie Dataset for open-world scenario for evaluation")
    # Point to the directory storing data
    dataset_dir = '../Dataset/OpenWorld/WalkieTalkie/'

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(dataset_dir + 'X_test_Mon_WalkieTalkie.pkl', 'rb') as handle:
        X_test_Mon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'y_test_Mon_WalkieTalkie.pkl', 'rb') as handle:
        y_test_Mon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'X_test_Unmon_WalkieTalkie.pkl', 'rb') as handle:
        X_test_Unmon = pickle.load(handle,encoding='iso-8859-1')
    with open(dataset_dir + 'y_test_Unmon_WalkieTalkie.pkl', 'rb') as handle:
        y_test_Unmon = pickle.load(handle,encoding='iso-8859-1')

    X_test_Mon = np.array(X_test_Mon)
    y_test_Mon = np.array(y_test_Mon)
    X_test_Unmon = np.array(X_test_Unmon)
    y_test_Unmon = np.array(y_test_Unmon)

    return X_test_Mon, y_test_Mon, X_test_Unmon, y_test_Unmon