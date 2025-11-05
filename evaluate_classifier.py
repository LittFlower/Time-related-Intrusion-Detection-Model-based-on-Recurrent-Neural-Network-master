import os
import numpy as np
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report


def main(time_steps: int = 1, batch_size: int = 1024):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    model_path = os.path.join(base_dir, 'models', 'best_model.hdf5')

    encoded_train = np.load(os.path.join(data_dir, 'encoded_train.npy'))
    train_label = np.load(os.path.join(data_dir, 'train_label.npy'))
    encoded_test = np.load(os.path.join(data_dir, 'encoded_test.npy'))
    test_label = np.load(os.path.join(data_dir, 'test_label.npy'))

    n_train = 172032
    n_test = 81920

    train = encoded_train[: (n_train + time_steps), :]
    train_label = train_label[: (n_train + time_steps), :]
    test = encoded_test[: (n_test + time_steps), :]
    test_label = test_label[: (n_test + time_steps), :]

    train_label_ = np.insert(train_label, 0, 0, axis=0)
    test_label_ = np.insert(test_label, 0, 0, axis=0)

    train_gen = TimeseriesGenerator(train, train_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)
    test_gen = TimeseriesGenerator(test, test_label_[:-1], length=time_steps, sampling_rate=1, batch_size=batch_size)

    model = load_model(model_path, compile=False)

    train_prob = model.predict(train_gen, verbose=1, steps=len(train_gen))
    test_prob = model.predict(test_gen, verbose=1, steps=len(test_gen))

    train_pred = (train_prob > 0.5)
    test_pred = (test_prob > 0.5)

    train_label_original = train_label_[(time_steps - 1) : -2, :]
    test_label_original = test_label_[(time_steps - 1) : -2, :]

    # Align lengths defensively
    t_len = min(len(train_pred), len(train_label_original))
    v_len = min(len(test_pred), len(test_label_original))
    train_pred = train_pred[:t_len]
    train_label_original = train_label_original[:t_len]
    test_pred = test_pred[:v_len]
    test_label_original = test_label_original[:v_len]

    np.save(os.path.join(data_dir, 'plot_prediction.npy'), test_pred)
    np.save(os.path.join(data_dir, 'plot_original.npy'), test_label_original)

    cm_train = confusion_matrix(train_label_original, train_pred)
    cm_test = confusion_matrix(test_label_original, test_pred)
    report = classification_report(test_label_original, test_pred)

    lines = []
    lines.append('Trainset Confusion Matrix')
    lines.append(str(cm_train))
    lines.append('Testset Confusion Matrix')
    lines.append(str(cm_test))
    lines.append('Classification Report')
    lines.append(report)
    output = "\n".join(lines)

    # echo to terminal
    print(output)

    # persist to file
    out_path = os.path.join(data_dir, 'eval_report.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(output + "\n")
    print(f"Saved evaluation report to: {out_path}")


if __name__ == '__main__':
    main()
