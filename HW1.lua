  -- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use: nb, logreg, hinge')
cmd:option('-logfile', 'log.txt', 'log file')
cmd:option('-test_weights', '', 'test on valid with weights from file')
cmd:option('-cross_val', 0, 'do cross validation')

-- Hyperparameters
cmd:option('-alpha', 1, 'alpha for naive Bayes')
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-batch_size', 50, 'batch size for SGD')
cmd:option('-max_epochs', 100, 'max # of epochs for SGD')
cmd:option('-lambda', 1.0, 'regularization lambda for SGD')

function train_nb(nclasses, nfeatures, X, Y, alpha)
  -- Trains naive Bayes model
  alpha = alpha or 0
  local N = X:size(1)
  local k = X:size(2)

  -- intercept
  local b = torch.histc(Y:double(), nclasses)
  local b_logsum = torch.log(b:sum())
  b:log():csub(b_logsum)

  local W = torch.Tensor(nclasses, nfeatures):fill(alpha)
  for i = 1, N do
    W:select(1, Y[i]):indexAdd(1, X[i]:long(), torch.ones(k))
  end
  -- zero out padding counts
  W:select(2, 1):zero()
  W:cdiv(W:sum(2):expand(W:size(1), W:size(2)))
  W:log()
  -- padding weight to zero
  W:select(2, 1):zero()

  return W, b
end

function NLL(pred, Y)
  -- Returns negative log-likelihood error.
  local N = Y:size(1)
  local err = 0
  for i = 1, N do
    err = err - torch.log(pred[i][Y[i]])
  end
  return err
end

function hinge(pred, Y)
  -- returns hinge error.
  local N = Y:size(1)
  local err = 0
  for i = 1, N do
    local val, ind = pred[i]:topk(2, 1, true)
    local x = 0
    if Y[i] == ind[1] then
      x = 1 - (pred[i][Y[i]] - val[2])
    else
      x = 1 - (pred[i][Y[i]] - val[1])
    end
    if x > 0 then
      err = err + x
    end
  end
  return err
end

function linear(X, W, b)
  -- performs y = softmax(x*W + b)
  local N = X:size(1)
  local z = torch.zeros(N, W:size(1))
  for i = 1, N do
    -- get predictions
    z[i] = W:index(2, X[i]:long()):sum(2)
    z[i]:add(b)
  end
  -- get softmax
  local max = z:max(2)
  local logsoftmax = z:clone()
  logsoftmax:csub(max:expand(logsoftmax:size(1), logsoftmax:size(2)))
  logsoftmax = torch.exp(logsoftmax):sum(2):log()
  logsoftmax:add(max:expand(logsoftmax:size(1), logsoftmax:size(2)))

  local Y_hat = z:clone()
  Y_hat:csub(logsoftmax:expand(Y_hat:size(1), Y_hat:size(2)))
  Y_hat:exp()
  return Y_hat, z
end

function cross_entropy_grad(X_batch, Y_batch, W, b)
    local N = X_batch:size(1)

    local Y_hat, _ = linear(X_batch, W, b)
    -- get gradient w.r.t. z
    local z_grad = Y_hat:clone()
    for i = 1, N do
      z_grad[i][Y_batch[i]] = Y_hat[i][Y_batch[i]] - 1
    end

    -- compute W, b grads
    local b_grad = z_grad:mean(1):squeeze()
    local W_grad = torch.zeros(nclasses, nfeatures)
    for i = 1, N do
        W_grad:indexAdd(2, X_batch[i]:long(), z_grad[i]:view(z_grad[i]:nElement(), 1):expand(nclasses, X_batch:size(2)))
    end
    W_grad:div(N)

    return W_grad, b_grad
end

function num_grad(X_batch, Y_batch, W, b, W_grad, b_grad, model)
    local batch_size = X_batch:size(1)
    local eps = 1e-5
    local del = torch.zeros(W:size(1), W:size(2))
    del[3][3] = eps
    local y1 = linear(X_batch, W + del, b)
    local y2 = linear(X_batch, W - del, b)

    local loss_f
    if model == 'logreg' then
      loss_f = NLL
    elseif model == 'hinge' then
      loss_f = hinge
    end
    print(X_batch)
    print('Actual W', (loss_f(y1, Y_batch) - loss_f(y2, Y_batch)) / (2 * eps * batch_size))
    print(W_grad[3][3])
    local yy1 = linear(X_batch, W, b + torch.Tensor{0,0,eps,0,0})
    local yy2 = linear(X_batch, W, b - torch.Tensor{0,0,eps,0,0})
    print('Actual b', (loss_f(yy1, Y_batch) - loss_f(yy2, Y_batch)) / (2 * eps * batch_size))
    print(b_grad[3])
    io.read()
end

function hinge_grad(X_batch, Y_batch, W, b)
  local N = X_batch:size(1)

  local Y_hat, _ = linear(X_batch, W, b)
  -- get gradient w.r.t. z
  local z_grad = torch.zeros(Y_hat:size(1), Y_hat:size(2))
  for i = 1, N do
      local _, ind = Y_hat[i]:topk(2, 1, true)
      local cprime
      if Y_batch[i] == ind[1] then
        cprime = ind[2]
      else
        cprime = ind[1]
      end
      local y_c = Y_hat[i][Y_batch[i]]
      local y_cprime = Y_hat[i][cprime]

      if y_c - y_cprime < 1 then
        z_grad[i] = z_grad[i]:mul(y_c - y_cprime)
        z_grad[i][Y_batch[i]] = z_grad[i][Y_batch[i]] - y_c
        z_grad[i][cprime] = z_grad[i][cprime] + y_cprime
      else
        z_grad[i]:zero()
      end
  end

  -- compute W, b grads
  local b_grad = z_grad:mean(1):squeeze()
  local W_grad = torch.zeros(nclasses, nfeatures)
  for i = 1, N do
      W_grad:indexAdd(2, X_batch[i]:long(), z_grad[i]:view(z_grad[i]:nElement(), 1):expand(nclasses, X_batch:size(2)))
  end
  W_grad:div(N)

  return W_grad, b_grad
end

function reg(W, lambda)
  return torch.pow(W, 2):sum() * lambda / 2
end

function train_reg(nclasses, nfeatures, X, Y, eta, batch_size, max_epochs, lambda, model)
  eta = eta or 0
  batch_size = batch_size or 0
  max_epochs = max_epochs or 0
  local N = X:size(1)

  -- initialize weights and intercept
  local W = torch.zeros(nclasses, nfeatures)
  local b = torch.zeros(nclasses)
  local epoch = 0

  local prev_loss = 1e10
  -- shuffle for batches
  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  Y = Y:index(1, shuffle)

  while epoch < max_epochs do
    -- loop through each batch
      for batch = 1, N, batch_size do
          if ((batch - 1) / batch_size) % 100 == 0 then
            print('Sample:', batch)
          end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          -- get gradients
          local W_grad, b_grad
          if model == 'logreg' then
            W_grad, b_grad = cross_entropy_grad(X_batch, Y_batch, W, b)
          elseif model == 'hinge' then
            W_grad, b_grad = hinge_grad(X_batch, Y_batch, W, b)
          end
          -- zero padding
          W_grad:select(2, 1):zero()

          -- numerical grads
          --num_grad(X_batch, Y_batch, W, b, W_grad, b_grad, model)

          -- regularization update
          W:mul(1 - eta * lambda * sz / N)
          -- update weights
          W:csub(W_grad:mul(eta))
          b:csub(b_grad:mul(eta))
          
          -- zero padding
          W:select(2, 1):zero()
      end

      -- calculate loss
      local pred = linear(X, W, b)
      local loss
      if model == 'logreg' then
        loss = NLL(pred, Y)
      elseif model == 'hinge' then
        loss = hinge(pred, Y)
      end
      loss = loss + reg(W, lambda)
      loss = loss / N
      print('Loss at epoch', epoch, ':', loss)

      if torch.abs(prev_loss - loss) / prev_loss < 0.001 then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save('train.t7', { W = W, b = b})
  end
  print('Trained', epoch, 'epochs')
  return W, b, prev_loss
end

function eval(X, Y, W, b, nclasses)
  -- Returns error from Y
  local pred, _ = linear(X, W, b)
  
  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()
  local err = argmax:eq(Y:long()):sum()
  err = err / Y:size(1)
  return argmax, err
end

function test()
  local X = torch.range(1, 12):reshape(3, 4)
  local W = torch.range(1, 24):reshape(2, 12)
  local b = torch.Tensor{1, 2}
  print('X W b')
  print(X, W, b)
  local y, z = linear(X, W, b)
  print('y z')
  print(y, z)
  io.read()
end

function cross_val(X, Y)
  -- 10 fold cross validation
  local N = X:size(1)

  local shuffle = torch.randperm(N):long()
  X = X:index(1, shuffle)
  Y = Y:index(1, shuffle)

  local fold_scores = {}
  for fold = 1, 10 do
      print('==> fold ', fold)
      -- split train/valid
      local i_start = math.floor((fold - 1) * 0.1 * N + 1)
      local i_end = math.floor(fold * 0.1 * N)
      local valid_X = X:narrow(1, i_start, i_end - i_start + 1)
      local valid_Y = Y:narrow(1, i_start, i_end - i_start + 1)
      local train_X = torch.cat(X:narrow(1, 1, i_start), X:narrow(1, i_end, N - i_end + 1), 1)
      local train_Y = torch.cat(Y:narrow(1, 1, i_start), Y:narrow(1, i_end, N - i_end + 1), 1)
    
      local _, err, loss = train(train_X, train_Y, valid_X, valid_Y)
      print('Fold', fold, 'error:', err)
      table.insert(fold_scores, err)
  end

  print('average CV score:', torch.Tensor(fold_scores):mean())
end

function train(X, Y, valid_X, valid_Y)
   local start = os.clock()

   -- Train.
   local W, b, loss
   if opt.classifier == 'nb' then
     W, b = train_nb(nclasses, nfeatures, X, Y, opt.alpha)
     loss = 'N/A'
   else
     -- sample for faster training
     --local batch_indices = torch.multinomial(torch.ones(X:size(1)), 10000, false):long()
     --X = X:index(1, batch_indices)
     --Y = Y:index(1, batch_indices)
     W, b, loss = train_reg(nclasses, nfeatures, X, Y, opt.eta, opt.batch_size, opt.max_epochs, opt.lambda, opt.classifier)
   end
   local time = os.clock() - start
   print('Training time:', time, 'seconds')
   print('Loss:', loss)

   -- Test.
   local pred, err = eval(valid_X, valid_Y, W, b, nclasses)
   print('Percent correct:', err)

   return pred, err, loss, time, W, b
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   nclasses = f:read('nclasses'):all():long()[1]
   nfeatures = f:read('nfeatures'):all():long()[1]

   print('Loading data...')
   local X = f:read('train_input'):all()
   local Y = f:read('train_output'):all()

   if opt.cross_val == 1 then
     print('Data loaded.')
     cross_val(X, Y)
   else 
     local valid_X = f:read('valid_input'):all()
     local valid_Y = f:read('valid_output'):all()
     local test_X = f:read('test_input'):all()
     print('Data loaded.')

     if opt.test_weights ~= '' then
       local A = torch.load('train.t7')
       local pred, err = eval(valid_X, valid_Y, A.W, A.b, nclasses)
       print('Percent correct:', err)
       os.exit()
     end

     local pred, err, loss, time = train(X, Y, valid_X, valid_Y, opt)

     -- Log results.
     f = io.open(opt.logfile, 'a')
     f:write(opt.classifier,' ',opt.alpha,' ',opt.eta,' ', opt.batch_size,' ', opt.max_epochs,' ', time,' ', loss,' ', err,' ',opt.lambda,'\n')
     f:close()
   end
end

main()
