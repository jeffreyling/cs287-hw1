-- Only requirement allowed
require("hdf5")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', 'SST1.hdf5', 'data file')
cmd:option('-classifier', 'nb', 'classifier to use')

-- Hyperparameters
cmd:option('-alpha', 1, 'alpha for naive Bayes')

function train_nb(nclasses, nfeatures, X, Y, alpha)
  alpha = alpha or 0
  local N = X:size(1)

  -- Trains naive Bayes model Y ~ X
  local b = torch.histc(Y:double(), nclasses)
  b:div(b:sum())
  b:log()

  local W = torch.Tensor(nclasses, nfeatures):fill(alpha)
  local indices = torch.linspace(1, X:size(1), X:size(1)):long()
  for c = 1, nclasses do
    W[c]:add(torch.histc(X:index(1, indices[Y:eq(c)]):double(), nfeatures))
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

function zero_one(pred, Y)
  -- Returns zero-one error
  local _, argmax = torch.max(pred, 2)
  return argmax:squeeze():eq(Y:long()):sum() / Y:size(1)
end

function eval(X, Y, W, b, model)
  -- Returns error from Y
  model = model or 'nb'

  local pred
  if model == 'nb' then
    pred = torch.zeros(X:size(1), W:size(1))
    for i = 1, X:size(1) do
      pred[i] = W:index(2, X[i]:long()):sum(2)
      pred[i]:add(b)
    end
  end

  -- Compute error from Y
  --local err = NLL(pred, Y)
  local err = zero_one(pred, Y)
  local _, argmax = torch.max(pred, 2)
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
   print('Data loaded.')

   --local W = torch.zeros(nclasses, nfeatures)
   --local b = torch.zeros(nclasses)

   -- Train.
   local W, b = train_nb(nclasses, nfeatures, X, Y, opt.alpha)
   print(W:narrow(2, 1, 10), b)

   -- Test.
   local pred, err = eval(valid_X, valid_Y, W, b)
   print(pred:narrow(1, 1, 10), err)
end

main()
