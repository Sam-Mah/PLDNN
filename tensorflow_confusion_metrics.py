import tensorflow as tf
import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, precision_score, recall_score, f1_score

def calculate_output(model, actual_classes, session, feed_dict):
    actuals = tf.argmax(actual_classes, 1)
    predictions = tf.argmax(model, 1)
    actuals = session.run(actuals, feed_dict)
    predictions = session.run(predictions, feed_dict)
    return actuals, predictions

def tf_confusion_metrics(model, actual_classes, session, feed_dict):
  import numpy as np
  cat = 5

  actuals, predictions = calculate_output(model, actual_classes, session, feed_dict)

  lbls = [*range(cat)]
  mcm = multilabel_confusion_matrix(actuals, predictions, labels=lbls)
  tp = mcm[:, 1, 1]
  tn = mcm[:, 0, 0]
  fn = mcm[:, 1, 0]
  fp = mcm[:, 0, 1]

  cm = confusion_matrix(actuals, predictions, labels=lbls, sample_weight=None)
  tp = np.mean(tp)
  tn = np.mean(tn)
  fp = np.mean(fp)
  fn = np.mean(fn)
  try:
      tpr = float(tp)/(float(tp) + float(fn))

      accuracy = (float(tp) + float(tn))/(float(tp) + float(fp) + float(fn) + float(tn))

      recall = tpr
      if((fp+tp)!=0):
        precision = float(tp)/(float(tp) + float(fp))
        f1_score = (2 * (precision * recall)) / (precision + recall)
      else:
        precision=0
        f1_score=0

      fp_rate=float(fp)/(float(fp)+float(tn))
      fn_rate=float(fn)/(float(fn)+float(tp))

      # return precision, recall, f1_score, accuracy, fp_rate, fn_rate
      PR = str(round(precision * 100, 2))
      RC = str(round(recall * 100, 2))
      F1 = str(round(f1_score * 100, 2))
      ACC = str(round(accuracy * 100, 2))
      FPR = str(round(fp_rate * 100, 2))
      FNR = str(round(fn_rate * 100, 2))

      data_pd=[['PR',PR],['RC', RC],['F1', F1],['ACC', ACC],['FPR', FPR], ['FNR', FNR],['tp', tp],['tn', tn],['fp', fp], ['fn', fn]]

      df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])

  except Exception as e:
    print(e)
    data_pd = [['PR', 'Err'], ['RC', 'Err'], ['F1', 'Err'], ['ACC', 'Err'], ['FPR', 'Err'], ['FNR', 'Err']]
    df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])

  return df


def tf_confusion_metrics_2(model, actual_classes, session, feed_dict):
    actuals, predictions = calculate_output(model, actual_classes, session, feed_dict)
    cm = tf.confusion_matrix(actuals, predictions)

    print("Confusion Matrix")

    return session.run(cm, feed_dict)

def Macro_calculate_measures_tf(y_true, y_pred, session, feed_dict):
    y_true, y_pred = calculate_output(y_pred, y_true, session, feed_dict)
    pr=  precision_score(y_true, y_pred, average='macro')
    rc = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    print("pr, rc, f1:" ,str(pr)+ str(rc)+str(f1))
    return pr, rc, f1
