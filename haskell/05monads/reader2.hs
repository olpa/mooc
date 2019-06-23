data Reader r a = Reader { runReader :: (r -> a) }

instance Applicative (Reader r) where
  pure = undefined
  (<*>) = undefined

instance Functor (Reader r) where
  fmap = undefined

instance Monad (Reader r) where
  return x = Reader $ \_ -> x
  m >>= k  = Reader $ \r -> runReader (k (runReader m r)) r

-- local' :: (r -> r') -> Reader r' a -> Reader r a
-- local' f m = Reader $ \e -> runReader m (f e)


type User = String
type Password = String
type UsersTable = [(User, Password)]

usersWithBadPasswords :: Reader UsersTable [User]
usersWithBadPasswords = do
  pairs <- Reader $ filter $ (== "123456") . snd
  return $ map fst pairs

t = runReader usersWithBadPasswords [("user", "123456"), ("x", "hi"), ("root", "123456")]
