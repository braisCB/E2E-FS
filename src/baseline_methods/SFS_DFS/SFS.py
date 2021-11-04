from tensorflow.keras import backend as K
import numpy as np
from src.baseline_methods.SFS_DFS import utils
from scipy.sparse import issparse


def get_saliency_func(data, label, saliency_function, batch_size=10, reduce_func='sum', generator=None,
                      generator_kwargs=None, generator_epochs=50):
    data_shape = data.shape[1:]
    ndata = len(data)
    if generator is not None:
        generator_kwargs = {} if generator_kwargs is None else generator_kwargs
        generator_iter = generator.flow(data, label, **generator_kwargs)
        steps_per_epoch = data.shape[0] // generator_kwargs['batch_size']
    if reduce_func is not None:
        saliency = np.zeros(data_shape)
        rfunction = getattr(np, reduce_func)
        if generator is not None:
            for r in range(generator_epochs):
                iter = 0
                while True:
                    gen_data, gen_label = next(generator_iter)
                    batch_saliency = saliency_function([gen_data, gen_label, 0])[0]
                    if saliency.shape != batch_saliency.shape:
                        batch_saliency = rfunction(batch_saliency, axis=0)
                    saliency = rfunction([saliency, batch_saliency], axis=0)
                    iter += 1
                    if iter == steps_per_epoch:
                        break
        else:
            index = 0
            while index < ndata:
                new_index = min(ndata, index + batch_size)
                batch_saliency = saliency_function([data[index:new_index], label[index:new_index], 0])[0]
                if saliency.shape != batch_saliency.shape:
                    batch_saliency = rfunction(batch_saliency, axis=0)
                saliency = rfunction([saliency, batch_saliency], axis=0)
                index = new_index
        # saliency /= max(1e-6, saliency.sum()) # np.linalg.norm(saliency, ord=2) #
    else:
        saliency = []
        if generator is not None:
            for r in range(generator_epochs):
                iter = 0
                while True:
                    gen_data, gen_label = next(generator_iter)
                    saliency.append(saliency_function([gen_data, gen_label, 0])[0])
                    iter += 1
                    if iter == steps_per_epoch:
                        break
        else:
            index = 0
            while index < ndata:
                new_index = min(ndata, index + batch_size)
                saliency.append(saliency_function([data[index:new_index], label[index:new_index], 0])[0])
                index = new_index
        saliency = np.concatenate(saliency, axis=0)
    return np.abs(saliency)


def get_saliency(
        data, label, saliency_function, batch_size=16, balance_data=False, type='classification', class_func='sum',
        reduce_func='sum', generator=None, generator_kwargs=None, generator_epochs=50, horizontal_flip=False
):
    data_ = data.toarray() if issparse(data) else np.asarray(data)
    if type == 'classification':
        saliency = get_saliency_for_classification(
            data_, label, saliency_function, batch_size=batch_size, balance_data=balance_data, class_func=class_func,
            reduce_func=reduce_func, generator=generator, generator_kwargs=generator_kwargs,
            generator_epochs=generator_epochs, horizontal_flip=horizontal_flip
        )
    else:
        saliency = get_saliency_for_regression(
            data_, label, saliency_function, batch_size=batch_size, balance_data=False, class_func=class_func,
            reduce_func=reduce_func, generator=generator, generator_kwargs=generator_kwargs,
            generator_epochs=generator_epochs, horizontal_flip=horizontal_flip
        )
    del data_
    return saliency


def get_saliency_for_classification(
        data, label, saliency_function, batch_size=50, balance_data=False, class_func='sum', reduce_func='sum',
        generator=None, generator_kwargs=None, generator_epochs=50, horizontal_flip=False
):
    label_ids = np.unique(label) if label.ndim == 1 else list(range(label.shape[-1]))
    saliency_results = []
    if isinstance(class_func, str):
        class_func = getattr(np, class_func)
    data_, label_ = data, label
    if balance_data:
        res = utils.balance_data(data, label)
        data_ = res[0]
        label_ = res[1]
    for label_id in label_ids:
        index = np.where(label_ == label_id)[0] if label_.ndim == 1 else np.where(label_[:, label_id] == 1)[0]
        label_saliency = get_saliency_func(data_[index], label_[index], saliency_function, batch_size, reduce_func,
                                           generator, generator_kwargs, generator_epochs)
        if horizontal_flip and len(data_.shape) > 2:
            label_saliency += get_saliency_func(
                data_[index][..., ::-1, :], label_[index], saliency_function, batch_size, reduce_func
            )
        label_saliency /= np.maximum(1e-6, np.sum(np.abs(label_saliency)))
        saliency_results.append(label_saliency)
    saliency = class_func(
        saliency_results,
        axis=0
    )
    # factor = np.sign(np.max(saliency_function, axis=0))
    # print('bad values :', (factor < 0).sum())
    saliency /= np.maximum(1e-6, np.abs(saliency).sum())
    del data_, label_, saliency_function
    return saliency


def get_saliency_for_regression(
        data, label, saliency_function, batch_size=50, balance_data=False, class_func='sum', reduce_func='sum',
        generator=None, generator_kwargs=None, generator_epochs=50, horizontal_flip=False
):
    saliency = get_saliency_func(data, label, saliency_function, batch_size, reduce_func, generator,
                                 generator_kwargs, generator_epochs)
    if horizontal_flip and len(data.shape) > 2:
        saliency += get_saliency_func(
            data[..., ::-1, :], label, saliency_function, batch_size, reduce_func
        )
    saliency /= max(1e-6, saliency.sum()) # np.linalg.norm(saliency, ord=2) #
    return saliency


def get_rank(fs_mode, data, label, model_func, valid_data=None, valid_label=None, type='classification',
             model_kwargs=None, fit_kwargs=None, saliency_kwargs=None,
             rank_kwargs=None, generator=None, generator_kwargs=None, return_info=False, **kwargs):
    data_shape = data.shape[1:]
    n_features = int(np.prod(data_shape))
    rank = np.arange(n_features).astype(int)
    reduce_data = True if len(data_shape) == 1 else False
    batch_size = 50
    epochs = 100
    verbose = 0
    model_kwargs = {} if model_kwargs is None else model_kwargs
    fit_kwargs_default = {} if generator is None else {
        'steps_per_epoch': len(data) // batch_size,
        'epochs': epochs,
        'verbose': verbose
    }
    fit_kwargs = utils.dict_merge(fit_kwargs_default, fit_kwargs)
    saliency_kwargs_default = {
        'balance_data': True,
        'batch_size': batch_size,
        'reduce_func': 'sum',
        'class_func': 'sum'
    }
    saliency_kwargs = utils.dict_merge(saliency_kwargs_default, saliency_kwargs)
    rank_kwargs_default = {
        'epsilon': 1,
        'reps': 1,
        'gamma': 0.9
    }
    rank_kwargs = utils.dict_merge(rank_kwargs_default, rank_kwargs)
    result = {
        'n_features': [],
        'accuracy': [],
    }
    while n_features > rank_kwargs['epsilon']:
        print('features =', n_features)
        if reduce_data:
            data_min = data[:, rank[:n_features]]
            if valid_data is not None:
                valid_data_min = valid_data[:, rank[:n_features]]
        else:
            mask = np.zeros(data_shape)
            mask.flat[rank[:n_features]] = 1.0
            data_min = data * mask
            if valid_data is not None:
                valid_data_min = valid_data * mask
        saliency = np.zeros(data_min.shape[1:])
        valid_acc = []
        for _ in range(rank_kwargs['reps']):
            model = model_func(data_min.shape[1:], **model_kwargs)
            if generator is None:
                model.fit(data_min, label, **fit_kwargs)
            else:
                generator_kwargs_default = {'batch_size': batch_size}
                generator_kwargs = utils.dict_merge(generator_kwargs_default, generator_kwargs)
                model.fit_generator(generator.flow(data_min, label, **generator_kwargs), **fit_kwargs)
            factor = 1. / model.evaluate(data_min, label, verbose=0)[0]
            if type == 'regression':
                factor = 1. / (factor + 1e-7)
            #     accuracy = 1.0
            # else:
            #     accuracy = (np.argmax(model.predict(data_min), axis=-1) == np.argmax(label, axis=-1)).sum() / len(label) \
            #         if label.ndim > 1 else (model.predict(data_min) == label).sum() / len(label)
            # pdata = data_min.copy()
            # print('init : ', model.evaluate(pdata, label, verbose=0))
            # for it in range(40):
            #     gradient = model.saliency_gradient([data_min, label, 0])[0]
            #     pdata += .01 * gradient
            #     print(it, ' : ', model.evaluate(pdata, label, verbose=0))
            if valid_data is not None:
                valid_acc.append(model.evaluate(valid_data_min, valid_label, verbose=0)[-1] * 100)
                print('acc : ', valid_acc[-1])
            print('factor : ', factor)
            if fs_mode.lower() == 'dfs':
                lasso_score = K.eval(K.abs(model.layers[1].kernel))
                saliency += factor * lasso_score / lasso_score.sum()
            elif fs_mode.lower() == 'sfs':
                saliency_function = model.saliency
                label_saliency = get_saliency(data_min, label, saliency_function, type=type,
                                              **saliency_kwargs)
                saliency += factor * label_saliency
            else:
                raise Exception('fs_mode not supported')
            del model
            K.clear_session()
        if not reduce_data:
            saliency = saliency.flatten()[rank[:n_features]]
        index = np.argsort(saliency)[::-1]
        rank[:n_features] = rank[index]
        if valid_data is not None:
            result['n_features'].append(n_features)
            result['accuracy'].append(valid_acc)
        n_features = int(rank_kwargs['gamma'] * n_features)
        # del data_min, valid_data_min, saliency
    if not return_info:
        return rank
    else:
        return {
            'rank': rank.tolist(),
            'classification': result
        }
