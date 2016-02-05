-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha', 19, 'alpha for naive Bayes')

function train_nb(nclasses, nfeatures, X, Y, alpha)
  -- Trains naive Bayes model
  alpha = alpha or 0
  local N = X:size(1)

  -- intercept
  local b = torch.histc(Y:double(), nclasses)
  b:div(b:sum())
  b:log()

  local W = torch.Tensor(nclasses, nfeatures):fill(alpha)
  local indices = torch.linspace(1, X:size(1), X:size(1)):long()
  for c = 1, nclasses do
    local counts = torch.histc(X:index(1, indices[Y:eq(c)]):double(), nfeatures)
    W[c]:add(counts)
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
  local probs = pred:index(2, Y:long())
  return -probs:log():sum()
end

function eval(X, Y, W, b, nclasses, model)
  -- Returns error from Y
  model = model or 'nb'

  local N = X:size(1)
  local pred = torch.zeros(N, nclasses)
  if model == 'nb' then
    for i = 1, N do
      pred[i] = W:index(2, X[i]:long()):sum(2)
      pred[i]:add(b)
    end
  end

  -- Compute error from Y
  local _, argmax = torch.max(pred, 2)
  argmax:squeeze()
  local err = argmax:eq(Y:long()):sum() / Y:size(1)
  return argmax, err
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
   local valid_X = f:read('valid_input'):all()
   local valid_Y = f:read('valid_output'):all()
   local test_X = f:read('test_input'):all()
   print('Data loaded.')

   --local W = torch.zeros(nclasses, nfeatures)
   --local b = torch.zeros(nclasses)

   -- Train.
   local W, b = train_nb(nclasses, nfeatures, X, Y, opt.alpha)

   -- Test.
   local pred, err = eval(valid_X, valid_Y, W, b, nclasses)
   print('Percent correct:', err)
end

main()
