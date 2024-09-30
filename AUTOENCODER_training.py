import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

import tensorflow_probability as tfp
import tensorflow as tf
import tensorflow.keras as tfk
import keras_lmu
import sys 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor

def mkdir(directory_path): 
    if os.path.exists(directory_path): 
        return directory_path
    else: 
        try: 
            os.makedirs(directory_path)
        except: 
            # in case another machine created the path meanwhile !:(
            return sys.exit("Erro ao criar diretório") 
        return directory_path
    
gpus=sys.argv[1]#"0"
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = gpus #gpus#'0,1,5'

if len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) > 1:
    strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
else:
    strategy = tf.distribute.get_strategy()
    

def prepare_data(train_path, test_path, zcol_name='Z', dataset=None, scaler=None):
    
    if not scaler:
        scaler = StandardScaler()
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # GRIZ
    keys = ['MAG_G', 'MAG_R', 'MAG_I', 'MAG_Z',
            'MAG_G-MAG_R', 'MAG_G-MAG_I', 'MAG_G-MAG_Z',
            'MAG_R-MAG_I', 'MAG_R-MAG_Z','MAG_I-MAG_Z']
    
    #GRI
    # keys = ['MAG_G', 'MAG_R', 'MAG_I',
    #         'MAG_G-MAG_R', 'MAG_G-MAG_I', 'MAG_R-MAG_I',]
    #GRZ
    # keys = ['MAG_G', 'MAG_R', 'MAG_Z',
    #         'MAG_G-MAG_R', 'MAG_G-MAG_Z', 'MAG_R-MAG_Z',]
    keys = [k.replace('MAG_', 'MAG_AUTO_') for k in keys]
    print(keys)
    y_train = train_data[zcol_name].values
    y_test = test_data[zcol_name].values
    x_train = train_data[keys].values
    x_test = test_data[keys].values

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)   

    print(f'X Train Shape = {x_train.shape}')
    print(f'Y Train Shape = {y_train.shape}')
    print(f'X Test Shape = {x_test.shape}')
    print(f'Y Test Shape = {y_test.shape}')

    return x_train, y_train, x_test, y_test, scaler

def create_model(num_components, params_size, inp_shape, n_pixels, event_shape = [1]):
    
    lmu_layer = keras_lmu.LMU(
                            memory_d=1,
                            order=128,
                            theta=n_pixels,
                            hidden_cell=tf.keras.layers.SimpleRNNCell(212),
                            hidden_to_memory=False,
                            memory_to_memory=False,
                            input_to_hidden=True,
                            kernel_initializer="glorot_normal",
                        )

    inp = tf.keras.Input((n_pixels, 1))
    x = lmu_layer(inp)

    x = tfk.layers.Dense(96, activation='relu')(x)
    x = tfk.layers.Dropout(0.2)(x)
    x = tfk.layers.BatchNormalization()(x)
    x = tfk.layers.Dense(96, activation='relu')(x)
    x = tfk.layers.Dropout(0.2)(x)
    x = tfk.layers.BatchNormalization()(x)

    
    x = tfk.layers.Dense(params_size)(x)

    x = tfp.layers.MixtureNormal(num_components, event_shape)(x)

    model = tfk.models.Model(inp, x)
    
    return model



def shuffle_idx(arr):
    shuffle_idx = np.random.choice(len(arr), len(arr), replace=False)
    
    return shuffle_idx

def cross_val_fit(X, Y, n_folds=5, random_state=42, 
                  save=True, save_dir='', batch_size = 512, 
                  epochs = 60, load_models=False,
                  num_components=20,params_size=60,event_shape=[1]):
    
    from sklearn.model_selection import KFold
    

    
    My_Models = {}
    My_Fits = {}
    
    fold = [f'fold_{j}' for j in range(n_folds)]
    
    i=0
    
    shuf_idx = shuffle_idx(X)
    
    X = X[shuf_idx]
    Y = Y[shuf_idx]
    
    if n_folds==1:
        
        val_cut = np.random.uniform(0,1, len(X))<.1
        
        val_idx = np.arange(len(X))[val_cut]
        train_idx = np.arange(len(X))[~val_cut]
        
        for train_idx, val_idx in [(train_idx,val_idx)]:

            mkdir(os.path.join(save_dir,fold[i]))

            best_loss_ckp = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir,fold[i],'best_model.h5'),
            monitor='val_loss',
            mode='auto',
            save_best_only=True)
            with strategy.scope():

                My_Models[fold[i]] = create_model(num_components=num_components, 
                                                  params_size=params_size,
                                                  inp_shape=X.shape[1:],
                                                  n_pixels=X.shape[1],
                                                  event_shape=event_shape)
                My_Models[fold[i]].compile(loss=lambda y, model: -model.log_prob(y), optimizer=tfk.optimizers.Nadam(learning_rate=2e-4))
                My_Models[fold[i]].summary()
                if load_models:
                    My_Models[fold[i]].load_weights(os.path.join(save_dir,fold[i],'best_model.h5'))

            if not load_models:
                My_Fits[fold[i]] = My_Models[fold[i]].fit(X[train_idx], Y[train_idx],
                                                        validation_data=(X[val_idx], Y[val_idx]),
                                                        verbose=1,
                                                        batch_size=batch_size,
                                                        epochs=epochs,
                                                        callbacks=[best_loss_ckp])
    
                with open(os.path.join(save_dir,fold[i],'history.pkl'), 'wb') as fp:
                    pickle.dump(My_Fits[fold[i]].history, fp)
                    print('dictionary saved successfully to file')


            print('\n')
            i+=1

        return My_Models, My_Fits
        
        
    else:
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kfold.split(X):

            mkdir(os.path.join(save_dir,fold[i]))

            best_loss_ckp = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(save_dir,fold[i],'best_model.h5'),
            monitor='val_loss',
            mode='auto',
            save_best_only=True)
            with strategy.scope():

                My_Models[fold[i]] = create_model(num_components=num_components, 
                                                  params_size=params_size,
                                                  inp_shape=X.shape[1:],
                                                  n_pixels=X.shape[1],
                                                  event_shape=event_shape)
                My_Models[fold[i]].compile(loss=lambda y, model: -model.log_prob(y), optimizer=tfk.optimizers.Nadam(learning_rate=2e-4))
                My_Models[fold[i]].summary()
                if load_models:
                    My_Models[fold[i]].load_weights(os.path.join(save_dir,fold[i],'best_model.h5'))

            if not load_models:
                My_Fits[fold[i]] = My_Models[fold[i]].fit(X[train_idx], Y[train_idx],
                                                        validation_data=(X[val_idx], Y[val_idx]),
                                                        verbose=1,
                                                        batch_size=batch_size,
                                                        epochs=epochs,
                                                        callbacks=[best_loss_ckp])
    
                with open(os.path.join(save_dir,fold[i],'history.pkl'), 'wb') as fp:
                    pickle.dump(My_Fits[fold[i]].history, fp)
                    print('dictionary saved successfully to file')

            print('\n')
            i+=1

        return My_Models, My_Fits

def loss_plot(Fits, initial_epoch=0, save_dir=None):
    
    all_train_losses = np.vstack([Fits[xn].history['loss'] for xn in Fits]).T
    all_val_losses = np.vstack([Fits[xn].history['val_loss'] for xn in Fits]).T

    epochs = all_val_losses.shape[0]
    
    train_loss_mean = np.mean(all_train_losses, axis=1)[initial_epoch:]
    val_loss_mean = np.mean(all_val_losses, axis=1)[initial_epoch:]

    train_loss_std = np.std(all_train_losses, axis=1)[initial_epoch:]
    val_loss_std = np.std(all_val_losses, axis=1)[initial_epoch:]


    plt.figure(figsize=(6,6))
    # mean plot
    plt.plot(range(initial_epoch, epochs),
            train_loss_mean,
            color='blue', label='train mean')
    plt.plot(range(initial_epoch, epochs),
            val_loss_mean,
            color='orange', label='validation mean')

    # std fill
    plt.fill_between(range(initial_epoch, epochs),
            train_loss_mean+train_loss_std,
            train_loss_mean-train_loss_std,
            color='blue',
            alpha=0.5)
    plt.fill_between(range(initial_epoch, epochs),
            val_loss_mean+val_loss_std,
            val_loss_mean-val_loss_std,
            color='orange',
            alpha=0.5)

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    
    if save_dir:
        
        plt.savefig( os.path.join(save_dir, 'loss.png') )
        
def generate_errors68(pdf, zaxis):
    
    zwidth = zaxis[1]-zaxis[0]
    cdfs = np.cumsum(pdf)*zwidth
    # zoffset=0.01
    p16 = np.sum(cdfs<.1585)*zwidth#np.percentile(pdfs, 15.85, axis=1)
    p84 = np.sum(cdfs<.8405)*zwidth#np.percentile(pdfs, 84.05, axis=1)
    
    err_68 = 0.5*(p84-p16)
    
    return err_68

def calc_PDF_series(weights, means, stds, x_range=None, optimize_zml=False):
    '''
    Returns a list of PDFs calculated as a combination of Gaussian functions

    Keyword arguments:
    x            -- Photometric redshift range for which the PDF should be calculated
    weights      -- Weight of the Gaussian components
    means        -- means of the Gaussian components
    stds         -- Standard deviation of the Gaussian components
    optimize_zml -- If the single-point estimate of photometric redshift should be optimized (if True, it will be
                    determined on a finer grid of points)
    '''
    
    if x_range is None:
        x = np.arange(-0.005, 1+0.001, 0.001) 
    else:
        x = x_range
                      
    # Convert columns from string to lists
    if type(weights) != np.ndarray:
        weights = np.array(weights)
        means   = np.array(means)
        stds    = np.array(stds)

    # Calculating PDFs and optimizing photo-zs (optional)
    PDFs           = []
    optimized_zmls = np.empty(len(means))
    
    if np.ndim(weights) == 2: # weights, means, and stds are 2D arrays 
        for i in tqdm(range(len(weights))):
            PDF = np.sum(weights[i]*(1/(stds[i]*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((x[:,None]-means[i])**2)/(stds[i])**2), axis=1)
            PDF = PDF#/np.trapz(PDF, x)
            PDFs.append(PDF)
        zmls = x[np.argmax(PDFs, axis=1)]
        
    if np.ndim(weights) == 1: # for one object
        PDF  = np.sum(weights*(1/(stds*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((x[:,None]-means)**2)/(stds)**2), axis=1)
        PDFs = PDF#/np.trapz(PDF, x)
        zmls = x[np.argmax(PDFs)]

    if optimize_zml == True:
        for i in tqdm(range(len(weights))):
            # First run
            optimized_x   = np.linspace(zmls[i]-0.002, zmls[i]+0.002, 500, endpoint=True)
            optimized_PDF = np.sum(weights[i]*(1/(stds[i]*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((optimized_x[:,None]-means[i])**2)/(stds[i])**2), axis=1)
            optimized_zml = optimized_x[np.argmax(optimized_PDF)]

            # Second run
            optimized_x   = np.linspace(optimized_zml-0.001, optimized_zml+0.001, 300, endpoint=True)
            optimized_PDF = np.sum(weights[i]*(1/(stds[i]*np.sqrt(2*np.pi))) * np.exp((-1/2) * ((optimized_x[:,None]-means[i])**2)/(stds[i])**2), axis=1)
            optimized_zmls[i] = optimized_x[np.argmax(optimized_PDF)]

        zmls = optimized_zmls
                
    return np.vstack(PDFs), zmls, x
    
    
def get_z_percentile(pdfs, zaxis, p):

    if np.ndim(pdfs) == 2:
        zwidth = zaxis[1]-zaxis[0]
        cdfs = np.cumsum(pdfs, axis=1)*zwidth

        zperc = np.sum(cdfs<p, axis=1)*zwidth#np.percentile(pdfs, 15.85, axis=1)
    
    else:
        zwidth = zaxis[1]-zaxis[0]
        cdfs = np.cumsum(pdfs)*zwidth

        zperc = np.sum(cdfs<p)*zwidth#np.percentile(pdfs, 15.85, axis=1)
     
        
    return zperc
    
    
    
    
def inference(My_Models, x_test, save_dir='', n_folds=4,
              n_obj=None, zaxis=np.linspace(0,2,2000),
              batch_pred=1024, num_components=20):
     
    My_Alphas = {}
    My_Mus = {}
    My_Sigmas = {}
    model_save_dir = save_dir
    if n_obj:
        new_x_test = x_test[:n_obj]
    else:
        new_x_test = x_test
    fold = [f'fold_{j}' for j in range(n_folds)]
    
    nsteps = int(len(new_x_test)/batch_pred)
    
    for j in range(n_folds):

        print('Getting Parameters --- ', fold[j].upper())
        My_Alphas[fold[j]] = np.zeros(shape=(len(new_x_test), num_components))
        My_Mus[fold[j]] = np.zeros(shape=(len(new_x_test), num_components))
        My_Sigmas[fold[j]] = np.zeros(shape=(len(new_x_test), num_components))

        for i in tqdm(range(nsteps)):

            if i == nsteps-1:
                gm = My_Models[fold[j]](new_x_test[i*batch_pred:])
                My_Alphas[fold[j]][i*batch_pred:] = gm.mixture_distribution.probs_parameter().numpy()
                My_Mus[fold[j]][i*batch_pred:] = np.squeeze(gm.components_distribution.mean().numpy())
                My_Sigmas[fold[j]][i*batch_pred:] = np.squeeze(np.sqrt(gm.components_distribution.variance().numpy()))
                # del(gm)
            else:
                gm = My_Models[fold[j]](new_x_test[i*batch_pred:(i+1)*batch_pred])
                My_Alphas[fold[j]][i*batch_pred:(i+1)*batch_pred] = gm.mixture_distribution.probs_parameter().numpy()
                My_Mus[fold[j]][i*batch_pred:(i+1)*batch_pred] = np.squeeze(gm.components_distribution.mean().numpy())
                My_Sigmas[fold[j]][i*batch_pred:(i+1)*batch_pred] = np.squeeze(np.sqrt(gm.components_distribution.variance().numpy()))
                # del(gm)


        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_alphas.npy', My_Alphas[fold[j]])
        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_mus.npy', My_Mus[fold[j]])
        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_sigmas.npy', My_Sigmas[fold[j]])
        

    
    My_PDFs = {} #np.zeros( (len(new_x_test), len(zaxis)) )
    My_Photoz = {} #np.zeros(len(new_x_test))
    My_Errors = {}
    My_Medians = {}
    for j in range(n_folds):


        print('Getting PDFs --- ',fold[j].upper())
        
        
        My_PDFs[fold[j]], My_Photoz[fold[j]], _ = calc_PDF_series(weights=My_Alphas[fold[j]],
                                                                means=My_Mus[fold[j]],
                                                                stds=My_Sigmas[fold[j]], 
                                                                x_range=zaxis, optimize_zml=True)
        
        
        

        My_Errors[fold[j]] = get_z_percentile(My_PDFs[fold[j]], zaxis, p=.8405) - \
                             get_z_percentile(My_PDFs[fold[j]], zaxis, p=.1585)
        
        My_Medians[fold[j]] = get_z_percentile(My_PDFs[fold[j]], zaxis, p=.5)


        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_pdfs.npy', My_PDFs[fold[j]])
        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_photoz.npy', My_Photoz[fold[j]])
        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_zerr.npy', My_Errors[fold[j]])
        np.save(mkdir(model_save_dir + fold[j] + '/') + 'my_medians.npy', My_Medians[fold[j]])
        
    return My_Photoz, My_PDFs, My_Alphas, My_Mus, My_Sigmas


def nnelu(input):
    """ Computes the Non-Negative Exponential Linear Unit
    """
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))

def build_autoencoder(n_components, feature_size=10):

    input_dim = (n_components, )


    input_alpha = tf.keras.layers.Input(input_dim)
    h1_alpha = tf.keras.layers.Dense(units=16, activation='linear')(input_alpha)

    input_mu = tf.keras.layers.Input(input_dim)
    h1_mu = tf.keras.layers.Dense(units=16, activation='linear')(input_mu)

    input_sigma = tf.keras.layers.Input(input_dim)
    h1_sigma = tf.keras.layers.Dense(units=16, activation='linear')(input_sigma)
    # h1_sigma = tf.keras.layers.Dense(units=16, activation='linear')(h1_sigma)
    
    
    concat_layer = tf.keras.layers.Concatenate()([h1_alpha, h1_mu, h1_sigma])

    h1_concat = tf.keras.layers.Dense(units=16, activation='linear')(concat_layer)
    
    
    
    latent_space = tf.keras.layers.Dense(units=feature_size, activation='linear')(concat_layer)
    encoder = tf.keras.models.Model(inputs =[input_alpha,input_mu,input_sigma], outputs = latent_space)

    
    
    # h2_concat = tf.keras.layers.Dense(units=16, activation='linear')(encoder.output)

    h2_alpha = tf.keras.layers.Dense(units=16, activation='linear')(encoder.output)
    h2_alpha = tf.keras.layers.Dense(units=n_components, activation='linear')(h2_alpha)
    output_alpha = tf.keras.layers.Dense(units=n_components, activation='softmax')(h2_alpha)

    h2_mu = tf.keras.layers.Dense(units=16, activation='linear')(encoder.output)
    h2_mu = tf.keras.layers.Dense(units=n_components, activation='softmax')(h2_mu)
    output_mu = tf.keras.layers.Dense(units=n_components, activation='relu')(h2_mu)

    h2_sigma = tf.keras.layers.Dense(units=16, activation='linear')(encoder.output)
    h2_sigma = tf.keras.layers.Dense(units=n_components, activation=nnelu)(h2_sigma)
    output_sigma = tf.keras.layers.Dense(units=n_components, activation=nnelu)(h2_sigma)

    output_layer =  tf.keras.layers.Concatenate()([output_alpha, output_mu, output_sigma])

    decoder = tf.keras.models.Model(inputs =latent_space, outputs = output_layer)

    autoencoder = tf.keras.models.Model(inputs=encoder.input, outputs=decoder(encoder.output))
    
    return encoder, decoder, autoencoder
    
def compile_model(n_components=20, feature_size=10):
    with strategy.scope():
            # model = create_model(x_train_lc.shape[1:], x_train_meta.shape[1:], classes.shape[0])
            # model.compile(optimizer=optimizers.RectifiedAdam(8e-4), loss='categorical_crossentropy', metrics=['accuracy'])
        encoder, decoder, autoencoder = build_autoencoder(n_components=n_components, feature_size=feature_size)
        autoencoder.compile(loss='mae', optimizer='Nadam')
        autoencoder.summary()
    return encoder, decoder, autoencoder

def train_AE(X, n_folds=1, n_components=20, feature_size=10,load_models=False, save_dir='', epochs=100, batch_size=512, random_state=42):
    
    # kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)    
    # splited_indexes = kfold.split(x_train, y_train)
    # splited_indexes
    
    My_Encoder = {}
    My_Decoder = {}
    My_Autoencoder = {}
    My_Fits_Autoencoder = {}
    
    fold = [f'fold_{j}' for j in range(n_folds)]

    i=0

    shuf_idx = shuffle_idx(X)
    
    X = X[shuf_idx]
    # Y = Y[shuf_idx]
    
    if n_folds==1:
        
        val_cut = np.random.uniform(0,1, len(X))<.0#################### CHANGED FROM .1
        
        val_idx = np.arange(len(X))[val_cut]
        train_idx = np.arange(len(X))[~val_cut]
        
        for train_idx, val_idx in [(train_idx,val_idx)]:

            if load_models:

                ####### TO BE DEFINED ##############
                pass #continue

            
            best_loss_ckp = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir+fold[i]+'/'+'best_autoencoder.h5',
            # save_weights_only=True,
            monitor='val_loss',
            mode='auto',
            save_best_only=True)


            train_alphas = X[:,:n_components]
            train_mus = X[:,n_components:n_components*2]
            train_sigmas = X[:,n_components*2:]

            
            
            My_Encoder[fold[i]], My_Decoder[fold[i]], My_Autoencoder[fold[i]] = compile_model(n_components=n_components, feature_size=feature_size)#(num_components, params_size, inp_shape=x_train.shape[1:], n_pixels=n_pixels)
            My_Fits_Autoencoder[fold[i]] = My_Autoencoder[fold[i]].fit(x=[train_alphas[train_idx], train_mus[train_idx], train_sigmas[train_idx]],
                                y=X[train_idx],
                                validation_data=([train_alphas[val_idx], train_mus[val_idx], train_sigmas[val_idx]],
                                                X[val_idx]),
                                batch_size=batch_size,
                                epochs=epochs)#,
                                # callbacks=[best_loss_ckp])
    
            My_Encoder[fold[i]].save(save_dir+fold[i]+'/'+'encoder.h5')
            My_Decoder[fold[i]].save(save_dir+fold[i]+'/'+'decoder.h5')
            My_Autoencoder[fold[i]].save(save_dir+fold[i]+'/'+'autoencoder.h5')
            
            with open(save_dir+fold[i]+'/'+'autoencoder_history.pkl', 'wb') as fp:
                pickle.dump(My_Fits_Autoencoder[fold[i]].history, fp)
            print('dictionary saved successfully to file')
    
            i+=1
    else:
        
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for train_idx, val_idx in kfold.split(X):

            if load_models:

                ####### TO BE DEFINED ##############
                pass

            
            best_loss_ckp = tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir+fold[i]+'/'+'best_autoencoder.h5',
            # save_weights_only=True,
            monitor='val_loss',
            mode='auto',
            save_best_only=True)
            
            My_Encoder[fold[i]], My_Decoder[fold[i]], My_Autoencoder[fold[i]] = compile_model()#(num_components, params_size, inp_shape=x_train.shape[1:], n_pixels=n_pixels)
            My_Fits_Autoencoder[fold[i]] = My_Autoencoder[fold[i]].fit(x=[scaled_Train_Alphas[train_idx], scaled_Train_Mus[train_idx], scaled_Train_Sigmas[train_idx]],
                                y=X_Train[train_idx],
                                validation_data=([scaled_Train_Alphas[val_idx], scaled_Train_Mus[val_idx], scaled_Train_Sigmas[val_idx]],
                                                X_Train[val_idx]),
                                batch_size=256,
                                epochs=100)#,
                                # callbacks=[best_loss_ckp])
    
            My_Encoder[fold[i]].save(model_save_dir+fold[i]+'/'+'encoder.h5')
            My_Decoder[fold[i]].save(model_save_dir+fold[i]+'/'+'decoder.h5')
            My_Autoencoder[fold[i]].save(model_save_dir+fold[i]+'/'+'autoencoder.h5')
            
            with open(model_save_dir+fold[i]+'/'+'autoencoder_history.pkl', 'wb') as fp:
                pickle.dump(My_Fits_Autoencoder[fold[i]].history, fp)
            print('dictionary saved successfully to file')
    
            i+=1
            
    return My_Encoder,My_Decoder,My_Autoencoder


class SigmaScaler():

    def __init__(self, stretch=2):
        self.stretch = stretch
        return 

    def fit_transform(self, data):
        return data*self.stretch

    def transform(self, data):
        return data*self.stretch

    def inverse_transform(self, data):
        return data/self.stretch


class MuScaler():
    def __init__(self):
        self.std_scaler = StandardScaler()
        return 

    def fit_transform(self, data):
        scaled_data = self.std_scaler.fit_transform(data.T)
        return scaled_data.T

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        unscaled_data = self.std_scaler.inverse_transform(data.T)
        return unscaled_data.T
    
class NoScaler():
    def __init__(self):
        return 

    def fit_transform(self, data):
        return data

    def transform(self, data):
        return data

    def inverse_transform(self, data):
        return data


def parameters_prepare(train_param_path, test_param_path,
                       alpha_scaler,
                       mu_scaler,
                       sigma_scaler):

    train_alphas = np.load(train_param_path+'my_alphas.npy')
    train_mus = np.load(train_param_path+'my_mus.npy')
    train_sigmas = np.load(train_param_path+'my_sigmas.npy')

    test_alphas = np.load(test_param_path+'my_alphas.npy')
    test_mus = np.load(test_param_path+'my_mus.npy')
    test_sigmas = np.load(test_param_path+'my_sigmas.npy')

    # alpha_scaler = NoScaler()
    # mu_scaler = NoScaler()
    # sigma_scaler = SigmaScaler(1)
    
    scaled_train_Alphas = alpha_scaler.fit_transform(train_alphas)
    scaled_train_Mus = mu_scaler.fit_transform(train_mus)
    scaled_train_Sigmas = sigma_scaler.fit_transform(train_sigmas)

    scaled_test_Alphas = alpha_scaler.transform(test_alphas)
    scaled_test_Mus = mu_scaler.fit_transform(test_mus)
    scaled_test_Sigmas = sigma_scaler.transform(test_sigmas)

    
    xtrain = np.concatenate([scaled_train_Alphas, scaled_train_Mus, scaled_train_Sigmas], axis=1)
    xtest = np.concatenate([scaled_test_Alphas, scaled_test_Mus, scaled_test_Sigmas], axis=1)

    return xtrain, xtest, alpha_scaler, mu_scaler, sigma_scaler


def inference_AE(Encoder, Decoder, Autoencoder, x_test,
                 alpha_scaler,
                 mu_scaler,
                 sigma_scaler,
                 save_dir='', n_folds=4,
                 n_obj=None, zaxis=np.linspace(0,2,2000),
                 batch_pred=1024, num_components=20):
     
    My_Alphas = {}
    My_Mus = {}
    My_Sigmas = {}
    My_Latent = {}
    # model_save_dir = save_dir
                  
    if n_obj:
        new_x_test = x_test[:n_obj]
    else:
        new_x_test = x_test
    fold = [f'fold_{j}' for j in range(n_folds)]
                  
    test_alphas = new_x_test[:,:num_components]
    test_mus = new_x_test[:,num_components:num_components*2]
    test_sigmas = new_x_test[:,num_components*2:]
    
    
    
    for j in range(n_folds):

        print('Getting Parameters --- ', fold[j].upper())

        



        My_Latent[fold[j]] = Encoder[fold[j]].predict([test_alphas,test_mus,test_sigmas])

        np.save(mkdir(save_dir + fold[j] + '/')+'my_latent.npy', My_Latent[fold[j]])

        ReconstructedParams = Decoder[fold[j]].predict(My_Latent[fold[j]])


        
        My_Alphas[fold[j]] = alpha_scaler.inverse_transform(ReconstructedParams[:,:num_components])#np.zeros(shape=(len(new_x_test), num_components))
        My_Mus[fold[j]] = mu_scaler.inverse_transform(ReconstructedParams[:,num_components:num_components*2])#np.zeros(shape=(len(new_x_test), num_components))
        My_Sigmas[fold[j]] = sigma_scaler.inverse_transform(ReconstructedParams[:,num_components*2:num_components*3])#np.zeros(shape=(len(new_x_test), num_components))


        np.save(mkdir(save_dir + fold[j] + '/') + 'my_alphas.npy', My_Alphas[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_mus.npy', My_Mus[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_sigmas.npy', My_Sigmas[fold[j]])
        

    
    My_PDFs = {} #np.zeros( (len(new_x_test), len(zaxis)) )
    My_Photoz = {} #np.zeros(len(new_x_test))
    My_Errors = {}
    My_Medians = {}
                  
    for j in range(n_folds):


        print('Getting PDFs --- ',fold[j].upper())
        
        
        My_PDFs[fold[j]], My_Photoz[fold[j]], _ = calc_PDF_series(weights=My_Alphas[fold[j]],
                                                                means=My_Mus[fold[j]],
                                                                stds=My_Sigmas[fold[j]], 
                                                                x_range=zaxis, optimize_zml=True)
        
        
        

        My_Errors[fold[j]] = get_z_percentile(My_PDFs[fold[j]], zaxis, p=.8405) - \
                             get_z_percentile(My_PDFs[fold[j]], zaxis, p=.1585)
        
        My_Medians[fold[j]] = get_z_percentile(My_PDFs[fold[j]], zaxis, p=.5)


        np.save(mkdir(save_dir + fold[j] + '/') + 'my_pdfs.npy', My_PDFs[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_photoz.npy', My_Photoz[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_zerr.npy', My_Errors[fold[j]])
        np.save(mkdir(save_dir + fold[j] + '/') + 'my_medians.npy', My_Medians[fold[j]])
        
    return My_Photoz, My_PDFs, My_Alphas, My_Mus, My_Sigmas


RESULTS_DIR = os.path.join(mkdir(f'results/'))

TRAIN_PATH = f'training_data/DELVE_DR2_train_flat_GRIZ.csv'
TEST_PATH = f'test_data/DELVE_DR2_test_flat_GRIZ.csv'

# setting nnelu function
tf.keras.utils.get_custom_objects().update({'nnelu': tf.keras.layers.Activation(nnelu)})

# loading parameters

TRAIN_PARAMS_PATH = 'training_data/fold_0/'
TEST_PARAMS_PATH = 'test_data/fold_0/'

ALPHA_SCALER = NoScaler()
MU_SCALER = NoScaler()
SIGMA_SCALER = SigmaScaler(1)

xtrain_params, xtest_params,alpha_scaler,mu_scaler,sigma_scaler = parameters_prepare(train_param_path=TRAIN_PARAMS_PATH,
                                   test_param_path=TEST_PARAMS_PATH,
                                   alpha_scaler=ALPHA_SCALER,
                                   mu_scaler=MU_SCALER,
                                   sigma_scaler=SIGMA_SCALER)

print('x_train shape => ',xtrain_params.shape)
print('x_test shape => ',xtest_params.shape)

#model_params
NUM_COMPONENTS=20
N_FOLDS=1
FEATURE_SIZE = 10 #SIZE OF THE LATTENT SPACE
EPOCHS = 200
BATCH_SIZE = 512*2
RANDOM_STATE = 137
ZAXIS=np.linspace(0,2,2000)
N_OBJ = None#1000 #None to predict over all dataset

#training autoencoder
My_Encoder,My_Decoder,My_Autoencoder = train_AE(X=xtrain_params, n_folds=N_FOLDS,
                                                n_components=NUM_COMPONENTS,
                                                feature_size=FEATURE_SIZE,load_models=False, save_dir=RESULTS_DIR, 
                                                epochs=EPOCHS, batch_size=BATCH_SIZE, random_state=RANDOM_STATE)


inference_AE(My_Encoder, My_Decoder, My_Autoencoder, xtest_params,
             alpha_scaler=alpha_scaler,
             mu_scaler=mu_scaler,
             sigma_scaler=sigma_scaler,
             save_dir=RESULTS_DIR, n_folds=N_FOLDS,
             n_obj=N_OBJ, zaxis=ZAXIS, num_components=NUM_COMPONENTS)





# '''
# RODAR TRAINING DE NOVO SALVANDO OS DADOS DE TREINO ASSIM COMO NO LEGACY

# fazer função que normalize os parametros
# .
# .
# .

# '''


# x_train, y_train, x_test, y_test, scaler = prepare_data(TRAIN_PATH, TEST_PATH,
#                                                         dataset=None, scaler=SCALER)





































# RESULTS_DIR = os.path.join(mkdir(f'/tf/astrodados/Results/DELVE_DR2_MDN_GRIZ/'))


# # MDN LOAD
# LOAD_MODELS = True

# #model_params
# NUM_COMPONENTS=20
# EVENT_SHAPE = [1]
# PARAMS_SIZE = tfp.layers.MixtureNormal.params_size(NUM_COMPONENTS, EVENT_SHAPE)




# #training
# EPOCHS = 200
# BATCH_SIZE = 512*2
# RANDOM_STATE = 137
# N_FOLDS = 1
# SCALER = StandardScaler()



# #Prediction
# ZAXIS=np.linspace(0,2,2000)
# BATCH_PRED = 4096
# N_OBJ = None#1000 #None to predict over all dataset



# tf.random.set_seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)

# x_train, y_train, x_test, y_test, scaler = prepare_data(TRAIN_PATH, TEST_PATH,
#                                                         dataset=None, scaler=SCALER)

# My_Models, My_Fits = cross_val_fit(x_train, y_train, n_folds=N_FOLDS,
#                                    random_state=RANDOM_STATE, save=True,
#                                    save_dir=RESULTS_DIR, batch_size=BATCH_SIZE,
#                                    epochs = EPOCHS, load_models=LOAD_MODELS,
#                                    num_components=NUM_COMPONENTS,
#                                    params_size=PARAMS_SIZE,
#                                    event_shape=EVENT_SHAPE)

# # loss_plot(My_Fits, initial_epoch=2, save_dir=RESULTS_DIR)

# inference(My_Models, x_test, save_dir=RESULTS_DIR, n_folds=N_FOLDS,
#           n_obj=N_OBJ, zaxis=ZAXIS,
#           batch_pred=BATCH_PRED,
#           num_components=NUM_COMPONENTS)