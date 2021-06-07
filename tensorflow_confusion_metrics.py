# from https://cloud.google.com/solutions/machine-learning-with-financial-time-series-data
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
  # predictions = tf.argmax(model, 1)
  # actuals = tf.argmax(actual_classes, 1)
  #
  # ones_like_actuals = tf.ones_like(actuals)
  # zeros_like_actuals = tf.zeros_like(actuals)
  # ones_like_predictions = tf.ones_like(predictions)
  # zeros_like_predictions = tf.zeros_like(predictions)
  #
  # tp_op = tf.reduce_sum(
  #   tf.cast(
  #     tf.logical_and(
  #       tf.equal(actuals, ones_like_actuals),
  #       tf.equal(predictions, ones_like_predictions)
  #     ),
  #     "float"
  #   )
  # )
  #
  # tn_op = tf.reduce_sum(
  #   tf.cast(
  #     tf.logical_and(
  #       tf.equal(actuals, zeros_like_actuals),
  #       tf.equal(predictions, zeros_like_predictions)
  #     ),
  #     "float"
  #   )
  # )
  #
  # fp_op = tf.reduce_sum(
  #   tf.cast(
  #     tf.logical_and(
  #       tf.equal(actuals, zeros_like_actuals),
  #       tf.equal(predictions, ones_like_predictions)
  #     ),
  #     "float"
  #   )
  # )
  #
  # fn_op = tf.reduce_sum(
  #   tf.cast(
  #     tf.logical_and(
  #       tf.equal(actuals, ones_like_actuals),
  #       tf.equal(predictions, zeros_like_predictions)
  #     ),
  #     "float"
  #   )
  # )
  #
  #
  # tp, tn, fp, fn = session.run(
  #     [tp_op, tn_op, fp_op, fn_op],
  #     feed_dict
  #   )
  #
  # acc = ''
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
      # fpr = float(fp)/(float(tp) + float(fn))

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

      # print ('Precision = ', precision)
      # print ('Recall = ', recall)
      # print ('F1 Score = ', f1_score)
      # print ('Accuracy = ', accuracy)
      # print ('FP Rate = ',fp_rate)
      # print ('FN Rate= ', fn_rate)

      # return precision, recall, f1_score, accuracy, fp_rate, fn_rate
      PR = str(round(precision * 100, 2))
      RC = str(round(recall * 100, 2))
      F1 = str(round(f1_score * 100, 2))
      ACC = str(round(accuracy * 100, 2))
      FPR = str(round(fp_rate * 100, 2))
      FNR = str(round(fn_rate * 100, 2))

      data_pd=[['PR',PR],['RC', RC],['F1', F1],['ACC', ACC],['FPR', FPR], ['FNR', FNR],['tp', tp],['tn', tn],['fp', fp], ['fn', fn]]

      df = pd.DataFrame(data_pd, columns=['Measure', 'Percentage'])


      # acc = 'PR: ' + PR + ' - RC: ' + RC + ' - F1: ' + F1 + ' - ACC: ' + ACC + ' - FPR: ' + FPR + ' - FNR: ' + FNR

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

