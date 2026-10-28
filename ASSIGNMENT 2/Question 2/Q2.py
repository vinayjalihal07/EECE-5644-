"""

Reproduces ML and Gaussian-prior MAP cubic-regression experiments using the
GMM generator shipped with the assignment. The fitted noise variance is a
plug-in estimate because the generator does not specify sigma squared.
"""
from pathlib import Path
import importlib.util
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import svd

ROOT = Path(__file__).resolve().parent
FIGS = ROOT / 'Q2_figures'
FIGS.mkdir(exist_ok=True)
SEED = 5644
N_TRAIN = 100
N_VALIDATE = 1000

def source_generator():
    src = ROOT / 'hw2q2.py'
    if not src.is_file():
        raise FileNotFoundError(f'Cannot find {src.name}. Put hw2q2.py in the same folder as this script.')
    spec = importlib.util.spec_from_file_location('provided_hw2q2_generator', src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generateData

def basis(x):
    x1 = np.asarray(x[:, 0], dtype=float)
    x2 = np.asarray(x[:, 1], dtype=float)
    return np.column_stack((np.ones(len(x1)),x1,x2,x1*x1,x1*x2,x2*x2,
                            x1**3,x1*x1*x2,x1*x2*x2,x2**3))

def mse(y, yhat):
    return np.mean((y - yhat)**2)

def main():
    np.random.seed(SEED)   # The supplied generator uses NumPy's global RNG.
    gen = source_generator()
    train = gen(N_TRAIN)
    validation = gen(N_VALIDATE)
    xtr, ytr = train[:2].T, train[2]
    xva, yva = validation[:2].T, validation[2]
    A = basis(xtr)
    B = basis(xva)
    U, s, Vt = svd(A, full_matrices=False)
    w_ml = Vt.T @ ((U.T @ ytr) / s)
    yhat_tr_ml = A @ w_ml
    sigma2_hat = mse(ytr, yhat_tr_ml)
    print('seed', SEED, 'train',len(ytr),'validate',len(yva))
    print('min-max xtrain',np.min(xtr,axis=0),np.max(xtr,axis=0))
    print('sigma2hat',sigma2_hat,'s min max',s[-1],s[0], 'condition',s[0]/s[-1])
    print('wML',w_ml)
    print('ML MSE train',mse(ytr,yhat_tr_ml),'val',mse(yva,B @ w_ml))

    gammas = np.logspace(-10, 12, 221)
    W = np.empty((len(gammas),len(w_ml)))
    train_mse = np.empty(len(gammas))
    val_mse = np.empty(len(gammas))
    w_norm = np.empty(len(gammas))
    dist_ml = np.empty(len(gammas))
    Uy = U.T @ ytr
    for i,gamma in enumerate(gammas):
        lam = sigma2_hat / gamma
        W[i] = Vt.T @ (s/(s*s + lam) * Uy)
        train_mse[i] = mse(ytr,A @ W[i])
        val_mse[i] = mse(yva,B @ W[i])
        w_norm[i] = np.linalg.norm(W[i])
        dist_ml[i] = np.linalg.norm(W[i]-w_ml)
    j = np.argmin(val_mse)
    best_gamma = gammas[j]
    best_w = W[j]
    print('Best gamma',best_gamma,'lambda',sigma2_hat/best_gamma,'MAP train',train_mse[j], 'val',val_mse[j], 'norm',w_norm[j])
    print('ML norm',np.linalg.norm(w_ml),'MAP vs ML gap',dist_ml[j])
    print('best coefficients',best_w)
    print('max likelihood', 'RMSE val',np.sqrt(mse(yva,B @ w_ml)))
    print('gamma grid selected:')
    for g in [1e-10,1e-8,1e-6,1e-4,1e-2,1,1e2,1e4,1e6,1e8,1e10,1e12]:
        k = np.argmin(abs(np.log10(gammas)-np.log10(g)))
        print(f'{gammas[k]:.1e} train {train_mse[k]:.6f} val {val_mse[k]:.6f} norm {w_norm[k]:.6f} distance {dist_ml[k]:.6f}')
    print('baseline zero predictor',mse(yva,np.zeros_like(yva)))

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10, 'axes.spines.top':False,'axes.spines.right':False, 'savefig.bbox':'tight', 'figure.dpi':135})
    fg, ax = plt.subplots(figsize=(8,4.4))
    ax.semilogx(gammas,val_mse,label='MAP validation ASE',lw=2.3,color='#177e93')
    ax.axhline(mse(yva,B@w_ml),label='ML validation ASE',ls='--',lw=2,color='#a65f21')
    ax.scatter([best_gamma],[val_mse[j]],color='#d1495b',s=75,zorder=5,label=f'Best on swept grid ({best_gamma:.2g})')
    ax.set(xlabel=r'Prior variance $\gamma$ (log scale)',ylabel='Validation average squared error',title='MAP validation error versus prior variance')
    ax.grid(alpha=.19)
    ax.legend(fontsize=8.6,frameon=False)
    fg.savefig(FIGS/'validation_ase_vs_gamma.png',dpi=215);plt.close(fg)

    fg, ax = plt.subplots(figsize=(8,4.4))
    ax.semilogx(gammas,val_mse,lw=2.4,color='#177e93',label='MAP validation ASE')
    ax.axhline(mse(yva,B@w_ml),ls='--',lw=2,color='#a65f21',label='ML validation ASE')
    ax.scatter([best_gamma],[val_mse[j]],s=70,color='#d1495b',zorder=5,label=f'Minimum on sweep = {val_mse[j]:.4f}')
    ax.set_xlim(1e-7,1e3)
    ax.set_ylim(4.1,6.0)
    ax.set(xlabel=r'Prior variance $\gamma$ (log scale)',ylabel='Validation average squared error',title='Validation error: detail around useful regularization')
    ax.grid(alpha=.20);ax.legend(frameon=False,fontsize=9)
    fg.savefig(FIGS/'validation_ase_zoom.png',dpi=215);plt.close(fg)

    fg, ax=plt.subplots(figsize=(8,4.4))
    ax.semilogx(gammas,train_mse,lw=2,color='#177e93',label='MAP training ASE')
    ax.semilogx(gammas,val_mse,lw=2,color='#c86728',label='MAP validation ASE')
    ax.axhline(mse(ytr,A@w_ml),ls=':',lw=1.5,color='#177e93',label='ML training ASE')
    ax.axhline(mse(yva,B@w_ml),ls=':',lw=1.5,color='#c86728',label='ML validation ASE')
    ax.set(xlabel=r'Prior variance $\gamma$ (log scale)',ylabel='Average squared error',title='Training versus validation error')
    ax.grid(alpha=.19);ax.legend(ncol=2,fontsize=8,frameon=False)
    fg.savefig(FIGS/'train_vs_validation_ase.png',dpi=215);plt.close(fg)

    fg, ax = plt.subplots(figsize=(8,4.4))
    ax.semilogx(gammas,w_norm,lw=2.3,color='#177e93',label=r'$\Vert\hat{w}_{MAP}\Vert_2$')
    ax.axhline(np.linalg.norm(w_ml),lw=2,ls='--',color='#a65f21',label=r'$\Vert\hat{w}_{ML}\Vert_2$')
    ax.set(xlabel=r'Prior variance $\gamma$ (log scale)',ylabel='Euclidean norm of coefficients',title='MAP regularization path')
    ax.grid(alpha=.19);ax.legend(frameon=False)
    fg.savefig(FIGS/'coefficient_norm_vs_gamma.png',dpi=215);plt.close(fg)

    fig,axs=plt.subplots(1,2,figsize=(9,4),sharex=True,sharey=True)
    for ax, yy, ttl in ((axs[0],B @ w_ml,'ML model'),(axs[1],B @ best_w,'Best MAP model')):
        ax.scatter(yva,yy,s=9,alpha=.4,c='#177e93',edgecolors='none')
        mn=min(np.min(yva),np.min(yy));mx=max(np.max(yva),np.max(yy))
        ax.plot([mn,mx],[mn,mx],ls='--',lw=1.5,color='#d1495b')
        ax.set_title(ttl)
        ax.set_xlabel('Observed validation y')
        ax.grid(alpha=.13)
    axs[0].set_ylabel('Predicted validation y')
    fig.suptitle('Observed versus fitted validation responses',fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS/'observed_vs_predicted.png',dpi=215);plt.close(fig)

    fg = plt.figure(figsize=(8,5.2));ax=fg.add_subplot(111,projection='3d')
    ax.scatter(xtr[:,0],xtr[:,1],ytr, s=14,alpha=.75,c=ytr,cmap='viridis')
    ax.set(xlabel='$x_1$',ylabel='$x_2$',zlabel='$y$',title='Provided generator: 100 training samples')
    ax.view_init(elev=22,azim=-67)
    fg.savefig(FIGS/'training_scatter3d.png',dpi=215);plt.close(fg)

    fg = plt.figure(figsize=(8,5.2));ax=fg.add_subplot(111,projection='3d')
    ax.scatter(xva[:,0],xva[:,1],yva, s=5,alpha=.37,c=yva,cmap='viridis')
    ax.set(xlabel='$x_1$',ylabel='$x_2$',zlabel='$y$',title='Provided generator: 1,000 validation samples')
    ax.view_init(elev=22,azim=-67)
    fg.savefig(FIGS/'validation_scatter3d.png',dpi=215);plt.close(fg)

    # Predictions at representative x values for a tangible contrast.
    obs = np.array([[ -10,0],[0,0],[10,0]],float)
    print('representative predictions rows x1=-10,0,10 x2=0; ML', basis(obs)@w_ml, 'MAP',basis(obs)@best_w)

    header = 'gamma,lambda,training_ASE,validation_ASE,coefficient_l2_norm,distance_to_ML'
    data = np.column_stack((gammas,sigma2_hat/gammas,train_mse,val_mse,w_norm,dist_ml))
    np.savetxt(ROOT/'Q2.csv', data, delimiter=',',header=header,comments='')
    return dict(seed=SEED,n_train=N_TRAIN,n_validate=N_VALIDATE, sigma2=sigma2_hat, cond=np.linalg.cond(A), w_ml=w_ml,
                ase_ml_train=mse(ytr,A@w_ml),ase_ml_val=mse(yva,B@w_ml), gamma_best=best_gamma,
                ase_map_best_val=val_mse[j],ase_map_best_train=train_mse[j],w_map_best=best_w,
                r2_ml=1-np.sum((yva-B@w_ml)**2)/np.sum((yva-yva.mean())**2),
                r2_best=1-np.sum((yva-B@best_w)**2)/np.sum((yva-yva.mean())**2))
if __name__ == '__main__':
    main()
