function[U, Vf, Vd, Vfc, Vdc, Loss] = MFOrth(R_train, K, lambda, lr, maxiter)

%%%%%%%Input-

%R-train: N*N*Nuser*Ntime*Nmod -> train data. N is # of node in brain
%networks, Nuser is # of subjects, Ntime is # of scans, Nmod is # of modality.
%Here, Nmod = 2.

%K: dimension of the latent space.

%lambda (struct type): lambda.lb1 -> alpha in Eq.9 (Total loss)
%                      lambda.lb2 = lambda.lb3 -> lambda_1 in Eq.9. Function and 
%                                                 structure network can have 
%                                                 different loss weights.
%                      lambda.lb4 -> lambda_2 in Eq.9. weight for regularization.  

%maxiter: max iteration.


%%%%%%Output-

%U: N*K*Nuser*Ntime -> coefficient matrix. 

%Vf and Vd: N*K*Nuser*Ntime -> The representation matrix of functional
%networks and structural networks respectively. For example, each
%subject has a new feature [Vf, Vd] for the future classification/prediction
%experiments.

%Vfc and Vdc: N*K*Ntime -> group consistent matrix of the brain networks at
%each scan.

lb1=lambda.lb1;
lb2=lambda.lb2;
lb3=lambda.lb3;
lb4=lambda.lb4;

nuser = size(R_train,3);
nnode = size(R_train,1);
ntime = size(R_train,4);

Xf=R_train(:,:,:,:,1);
Xd=R_train(:,:,:,:,2);
U = rand(nnode, K, nuser, ntime);
Vf = rand(nnode, K, nuser, ntime);
Vd = rand(nnode, K, nuser, ntime);
Vfc = rand(nnode, K, ntime);
Vdc = rand(nnode, K, ntime);
%Xd_avg=zeros(nnode,nnode,ntime);
tmp_Xd=zeros(nnode,nnode);
for i=1:ntime
    for j=1:nuser
        tmp_Xd=tmp_Xd+Xd(:,:,j,i);
    end
    %Xd_avg(:,:,i)=tmp_Xd./nuser;
    [E, psd] = chol(eye(nnode));
    E_inv = inv(E);
    Vdc(:,:,i) = E_inv(:, end-K+1:end);
end


% objective function handle.
f = @(vd, vdc) fVdc(vd, vdc, nuser, lb3, lb4);
% gradient function handle.
grad = @(vd, vdc, K) partial_Vdc(vd, vdc, K, nuser, lb3, lb4);

for i=1:ntime
    M(:,:,i) = eye(nnode);
end


%%%%%%% Hyperparameters Group 1. Ref: Jae Hwang, et al., 2015.  A projection 
%%%%%%% free method for generalized eigenvalue problem with a
%%%%%%% nonsmoothregularizer.

opts = struct();
opts.rho1 = 0.001;           % Wolfe condition c1 - step length (Armijo)
opts.rho2 = 0.999;           % Wolfe condition c2 - curvature
opts.grad_epsilon = 1e-7;   % terminate when gradient change less than epsilon
opts.print_interval = 200;  % print outputs interval
opts.max_iter = 1500;      % maximum iteration
opts.k = K;
%%%%%%
Loss=[];


%% start training
iter0 = 0;
loss_diff=10000;
while(iter0 < maxiter)
    
    iter0 = iter0 + 1;
    
    for i=1:ntime
        
        vfc=Vfc(:,:,i);
        vfc_grad=0.*vfc;
        vdc=Vdc(:,:,i);
        
        for j=1:nuser
            
            u=U(:,:,j,i);
            vf=Vf(:,:,j,i);
            vd=Vd(:,:,j,i);
            xf=Xf(:,:,j,i);
            xd=Xd(:,:,j,i);
            
            % Update Ufd
            u=u-2*lr*((u*(vf')*vf-xf*vf)+lb1*(u*(vd')*vd-xd*vd)+lb4*u);
            U(:,:,j,i)=u;
            
            %Update vf
            vf=vf-2*lr*((vf*(u')*u-xf*u)+lb2*(vf-vfc)+lb4*vf);
            Vf(:,:,j,i)=vf;
            
            %Update vd
            vd=vd-2*lr*(lb1*(vd*(u')*u-xd*u)+lb3*(vd-vdc)+lb4*vd);
            Vd(:,:,j,i)=vd;
            
            %Update vfc step1
            vfc_grad=vfc_grad+2*lb2*(vfc-vf);
            
        end
        
        %Update vfc step2
        vfc=vfc-lr*(vfc_grad+2*lb4*vfc);
        Vfc(:,:,i)=vfc;
    end
    
    iter1=0;
    pre_eng=0;
    cur_eng=1000;
    
    if(iter0<2) 
        maxiter_RGEP=1;
        opts.max_iter = 400;
    else
        maxiter_RGEP=1;
        opts.max_iter = 100;
    end
        
    while iter1<maxiter_RGEP && abs(pre_eng-cur_eng)>1e-1
        iter1=iter1+1;
        pre_eng=cur_eng;
        cur_eng=0;
        for i=1:ntime
            vdc=Vdc(:,:,i);
            %Update Vdc using RGEP
            vd=Vd(:,:,:,i);
            Md=M(:,:,i);
            vdc = RGEP(vd, f, grad, Md, vdc, opts);
            Vdc(:,:,i)=vdc;
            cur_eng=cur_eng+f(vd,vdc);
        end
        
        fprintf('...Previous RGEP Energy is: %d; Current RGEP Energy is: %d... \n', pre_eng, cur_eng);
        
        %%% Update R matrix
        R=zeros(nnode,nnode,ntime,ntime);
        for i=1:ntime
            for j=i:ntime
                [derror,Ytrains,transform]=procrustes(Vdc(:,:,i)',Vdc(:,:,j)','reflection',false);
                R(:,:,j,i)=transform.T;
                R(:,:,i,j)=R(:,:,j,i)';
            end
        end
        
        %%% Update M matrix
        M=zeros(nnode,nnode,ntime);
        for i=1:ntime
            if i==1
                M(:,:,i)=R(:,:,i+1,i)*Vdc(:,:,i+1)*Vdc(:,:,i+1)'*R(:,:,i+1,i)';
            elseif i==ntime
                M(:,:,i)=R(:,:,i-1,i)*Vdc(:,:,i-1)*Vdc(:,:,i-1)'*R(:,:,i-1,i)';
            else
                M(:,:,i)=R(:,:,i-1,i)*Vdc(:,:,i-1)*Vdc(:,:,i-1)'*R(:,:,i-1,i)';
                M(:,:,i)=M(:,:,i)+R(:,:,i+1,i)*Vdc(:,:,i+1)*Vdc(:,:,i+1)'*R(:,:,i+1,i)';
                M(:,:,i)=0.5*M(:,:,i);
            end
        end
        
    end
    
    for i=1:ntime
        for j=1:nuser
            Loss_t= norm(Xf(:,:,j,i)-U(:,:,j,i)*(Vf(:,:,j,i)'),'fro').^2;
            Loss_t= Loss_t + lb1*norm(Xd(:,:,j,i)-U(:,:,j,i)*(Vd(:,:,j,i)'),'fro').^2; 
            Loss_t= Loss_t + lb2*norm(Vf(:,:,j,i)-Vfc(:,:,i),'fro').^2;
            Loss_t= Loss_t + lb3*norm(Vd(:,:,j,i)-Vdc(:,:,i),'fro').^2;
            Loss_t= Loss_t + lb4*norm(U(:,:,j,i),'fro').^2;
            Loss_t= Loss_t + lb4*(norm(Vd(:,:,j,i),'fro').^2+norm(Vf(:,:,j,i),'fro').^2);
        end
        Loss_t = Loss_t + lb4*(norm(Vdc(:,:,i),'fro').^2+norm(Vfc(:,:,i),'fro').^2);
    end
    Loss(iter0)=Loss_t;
    
    %fprintf(fileID,'%d %6.3\n',iter0,Loss_t);
    
    if iter0>1
        fprintf('......\n It is the %d -th iteration. Previous Loss: %d; Current Loss is: %d\n...... \n', iter0, ...
        Loss(iter0-1), Loss(iter0));
        loss_diff= Loss(iter0) - Loss(iter0-1);
        if   (Loss(iter0-1) < Loss(iter0)|| abs(loss_diff/Loss(iter0))<0.01)
            break;
        end
    end
end
%fclose(fileID);















