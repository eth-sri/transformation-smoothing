from classifyHeuristic import *
from scipy.stats import binom_test

def smooth_pred(args, model, img, delta=None):
    counts =  sample(args, model, img, args.n_gamma, delta=delta).most_common(2)
    C0, cnt0 = counts[0]
    C1, cnt1 = counts[1] if len(counts) > 1 else (None, 0)
    if binom_test(cnt0, cnt0 + cnt1, p=0.5) > args.alpha_gamma:
        return None
    else:
        return C0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = setup_args(parser)
    parser = setup_args_preprocessing(parser)
    args = parser.parse_args()
    setup(args)
    model = get_basemodel(args)
    data = get_data(args)
    logger = get_logger(args, __file__)
    print(args, file=logger)
    print("index", "label", "predicted", "radius", "abstain", "incorrect", "abstain/wo/int", "incorrect/wo/int", file=logger)    
    for i, d in enumerate(data):
        img, label = d
        pred = classify(args, model, img)
        pred_s, R = smooth(args, model, img)
        if R > 0:
            if args.transformation == 'rot':
                angles = np.random.uniform(-R, R, args.attack_k)
                images = zip(getRotations(img, angles), angles)
            elif args.transformation == 'trans':
                offsets = []
                while len(offsets) < args.attack_k:
                    offset = np.random.uniform(-R, R, (2, 1))
                    l2 = np.sqrt((offset*offset).sum())
                    if l2 <= R:
                        offsets.append(offset)
                offsets_np = np.concatenate(offsets, axis=1)
                images = zip(getTranslations(img, offsets_np), offsets)
            cnt_abstain = 0
            cnt_false = 0
            cnt_abstain_wo_interpolation = 0
            cnt_false_wo_interpolation = 0
            for img_a, delta in images:
                # with interpolation
                pred_a_s = smooth_pred(args, model, img_a)
                if pred_a_s is None:
                    cnt_abstain += 1
                elif pred_a_s != pred_s:
                    cnt_false += 1
                # without interpolation
                # pred_a_s_wo_interpolation = smooth_pred(args, model, img, delta=delta)
                # if pred_a_s_wo_interpolation is None:
                #     cnt_abstain_wo_interpolation += 1
                # elif pred_a_s_wo_interpolation != pred_s:
                #     cnt_false_wo_interpolation += 1                
            print(i, label, pred_s, R, cnt_abstain, cnt_false,
                  cnt_abstain_wo_interpolation, cnt_false_wo_interpolation, file=logger)
