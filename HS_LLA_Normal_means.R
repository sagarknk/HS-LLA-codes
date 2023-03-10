rm(list=ls())
set.seed(100)

HS_LLA <- function(Y_mat, tau){
  
  p= length(Y_mat)
  theta_vec_init = rep(1,p)
  theta_vec_current = theta_vec_init
  theta_vec_next = rep(0,p)
  step_size = 1e-2
  dawson_vals = read.csv("dawson_vals.csv", header = FALSE)
  dawson_vals = as.matrix(dawson_vals)
  dawson_vals = as.numeric(dawson_vals)
  reduced_dawson_vals = dawson_vals[1:3501]
  u_grid = seq(0,2e3,step_size)
  reduced_u_grid = u_grid[1:3501]
  pre_saved_penalty = sum(u_grid*dawson_vals/tau)/sum(dawson_vals)
  norm_diff = sqrt(sum((theta_vec_next - theta_vec_current)^2))
  iter =1
  penalty_vec = rep(NA, p)
  while (norm_diff>1e-3){
    #cat(paste0("Iter number: ",iter,". Norm diff is: ",norm_diff,"\n"))
    theta_vec_current = theta_vec_init
    for(theta_idx in c(1:p)){
      
      if(theta_vec_init[theta_idx]!=0){
        
        penalty_num = (reduced_u_grid/tau)*exp(-(abs(theta_vec_init[theta_idx])*reduced_u_grid/tau))*reduced_dawson_vals
        penalty_denom = exp(-(abs(theta_vec_init[theta_idx])*reduced_u_grid/tau))*reduced_dawson_vals
        
        penalty = sum(penalty_num)/sum(penalty_denom)
        penalty_vec[theta_idx] = penalty
       
        theta_hat = -Y_mat[theta_idx]
        if (theta_hat < -penalty){
          theta_vec_init[theta_idx] = (-1*theta_hat - penalty)
        }else if (theta_hat > penalty){
          theta_vec_init[theta_idx] = (-1*theta_hat + penalty)
        }else{
          theta_vec_init[theta_idx] = 0
        }
      }else{
        
        penalty = pre_saved_penalty
        penalty_vec[theta_idx] = penalty
        
        theta_hat = -Y_mat[theta_idx]
        if (theta_hat < -penalty){
          theta_vec_init[theta_idx] = (-1*theta_hat - penalty)
        }else if (theta_hat > penalty){
          theta_vec_init[theta_idx] = (-1*theta_hat + penalty)
        }else{
          theta_vec_init[theta_idx] = 0
        }
      }
      
    }
    theta_vec_next = theta_vec_init
    norm_diff = sqrt(sum((theta_vec_next - theta_vec_current)^2))
    iter = iter+1
  }
  return(list(theta_est = theta_vec_next, num_iter = iter-1))
}
##########################################################
HS_LLA_folds <- function(Y_mat, n,tau,tau_idx)
{
  library(cvTools)
  num_folds = 10
  fold_idx = cvFolds(n,num_folds, type="consecutive")
  prediction_MSE_this_tau = rep(0, num_folds)
  for(fold_iter in 1:num_folds)
  {

    y_train = y[fold_idx$which!=fold_iter]
    y_test = y[fold_idx$which==fold_iter]

    eval_HS_LLA = HS_LLA(y_train,tau)
    coef_HS_LLA = eval_HS_LLA$theta_est
    
    Yhat_HS_LLA=rep(mean(coef_HS_LLA), n/num_folds)

    prediction_MSE_this_tau[fold_iter] = sum((y_test - Yhat_HS_LLA)^2)
  }
  #cat(paste0("Tau_idx = ", tau_idx," finished","\n"))
  return(mean(prediction_MSE_this_tau))
}
#################################################
library(foreach)
library(doParallel)
library(sparsenet)
library(glmnet)
library(ncvreg)
library(cvTools)
library(horseshoe)

num_folds = 10
n = 50
fold_idx = cvFolds(n,num_folds, type="consecutive")
c1= makeCluster(detectCores()-1)
registerDoParallel(c1)
tau_seq = exp(seq(-3,0,0.01))
num_data_sets = 50
All_data_sets_SSE = matrix(0, num_data_sets, 5)
All_data_sets_pSSE = matrix(0, num_data_sets, 5)
All_data_sets_TN = matrix(0, num_data_sets, 5)
All_data_sets_TP = matrix(0, num_data_sets, 5)
All_data_sets_time = matrix(0, num_data_sets, 5)
#################################################

for(data_idx in 1:num_data_sets)
{
  n=50
  theta= c(3,3, rep(0,n-2))
  y=rnorm(n=n,mean=theta,sd=1)
  y_out_of_sample=rnorm(n=n,mean=theta,sd=1)
  ###############################################
  prediction_avg_MSE = foreach(tau_idx = 1:length(tau_seq), .combine=cbind)%dopar%{
    prediction_MSE_this_tau = HS_LLA_folds(y,n,tau_seq[tau_idx],tau_idx)
    prediction_MSE_this_tau
  }
  ###################################
  ##Fit HS-LLA
  tau=tau_seq[which(prediction_avg_MSE==min(prediction_avg_MSE))]
  Y_mat=y
  ptm<-proc.time()
  eval_HS_LLA = HS_LLA(y,tau)
  t_HS_LLA<-proc.time() - ptm
  coef_HS_LLA = eval_HS_LLA$theta_est
  nzero_HS_LLA=length(which(coef_HS_LLA==0 & theta==0))
  out_of_sample_HS_LLA = sum((y_out_of_sample - coef_HS_LLA)^2)
  ###################################
  ##Fit LASSO
  ptm<-proc.time()
  LASSOfit=cv.glmnet(x=diag(n), y=y, foldid = fold_idx$which)
  tLASSO<-proc.time() - ptm
  coefLASSO=coef(LASSOfit,s="lambda.min")
  coefLASSO=coefLASSO[-1]
  out_of_sample_LASSO = sum((y_out_of_sample - coefLASSO)^2)
  ###################################
  ##Fit MCP
  ptm<-proc.time()
  MCPfit=cv.sparsenet(x=diag(n), y=y, foldid = fold_idx$which)
  tMCP<-proc.time() - ptm
  coefMCP=coef(MCPfit)
  coefMCP=coefMCP[-1]
  out_of_sample_MCP = sum((y_out_of_sample - coefMCP)^2)
  ###################################
  ##Fit SCAD
  ptm<-proc.time()
  SCADfit=cv.ncvreg(X=diag(n), y=y, penalty="SCAD",foldid = fold_idx$which)
  tSCAD<-proc.time() - ptm
  coefSCAD=coef(SCADfit)
  coefSCAD=coefSCAD[-1]
  out_of_sample_SCAD = sum((y_out_of_sample - coefSCAD)^2)
  #################################################
  ##Fit HS MCMC
  ptm<-proc.time()
  evalHS=HS.normal.means(y=y, method.tau ="halfCauchy",burn = 5000, nmc = 10000, alpha = 0.25)
  tHS<-proc.time() - ptm
  coefHS=evalHS$BetaHat
  Yhat_HS= coefHS
  out_of_sample_HS = sum((Yhat_HS - y_out_of_sample)^2)
  coefHS_variable_select = (evalHS$LeftCI <0) & (evalHS$RightCI>0)
  coefHS_variable_select = !coefHS_variable_select
  coefHS_variable_select = coefHS_variable_select*1
  ###############################################
  mse_HS_LLA=sum((theta-coef_HS_LLA)^2)
  mse_LASSO=sum((theta-coefLASSO)^2)
  mse_MCP=sum((theta-coefMCP)^2)
  mse_SCAD=sum((theta-coefSCAD)^2)
  mse_HS=sum((theta-coefHS)^2)
  ###############################################
  All_data_sets_SSE[data_idx,] = c(round(mse_HS_LLA,2), round(mse_SCAD,2), 
                                   round(mse_MCP,2), round(mse_LASSO,2), round(mse_HS,2))
  All_data_sets_pSSE[data_idx,] = c(round(out_of_sample_HS_LLA,2), round(out_of_sample_SCAD,2), 
                                    round(out_of_sample_MCP,2), round(out_of_sample_LASSO,2), round(out_of_sample_HS,2))
  ###############################################
  HS_LLA0=length(which(coef_HS_LLA==0 & theta==0))
  MCP0=length(which(coefMCP==0 &theta==0))
  SCAD0=length(as.vector(which(coefSCAD==0 &theta==0)))
  LASSO0=length(which(coefLASSO==0 & theta==0))
  HS0 = length(which(coefHS_variable_select==0 & theta==0))
  ###############################################
  All_data_sets_TN[data_idx,] = c(HS_LLA0,SCAD0,MCP0,LASSO0, HS0)
  ###############################################
  HS_LLA1=length(which(coef_HS_LLA!=0 & theta!=0))
  MCP1=length(which(coefMCP!=0 &theta!=0))
  SCAD1=length(as.vector(which(coefSCAD!=0 &theta!=0)))
  LASSO1=length(which(coefLASSO!=0 & theta!=0))
  HS1 =  length(which(coefHS_variable_select!=0 & theta!=0))
  ###############################################
  All_data_sets_TP[data_idx,] = c(HS_LLA1,SCAD1,MCP1,LASSO1,HS1)
  ###############################################
  All_data_sets_time[data_idx,] = c(t_HS_LLA[[1]],tSCAD[[1]], tMCP[[1]], tLASSO[[1]],tHS[[1]])
  ###############################################
  cat(paste0("Data number ", data_idx," is done","\n"))
}

stopCluster(c1)

round(colMeans(All_data_sets_SSE),3)
round(apply(All_data_sets_SSE, 2, sd),3)

round(colMeans(All_data_sets_pSSE),3)
round(apply(All_data_sets_pSSE, 2, sd),3)

round(colMeans(All_data_sets_TN)/sum(theta==0),3)
round(apply(All_data_sets_TN/sum(theta==0), 2, sd),3)

round(colMeans(All_data_sets_TP)/sum(theta!=0),3)
round(apply(All_data_sets_TP/sum(theta!=0), 2, sd),3)

round(colMeans(All_data_sets_time),2)


