/******************************************************************************
Class to handle our own exceptions for the tf2gap8 code
Contributors: Corine Lamagdeleine
*******************************************************************************/

#include <iostream>
#include <signal.h>
#include <stdio.h>
#include <unistd.h>

using namespace std;
#include <exception>

class tf2gap8Exception: public exception
{
  public:
  tf2gap8Exception(const string& msg) : msg_(msg) {}
  virtual const char* what() const throw()
  {
    return msg_.c_str();
  }
  string msg_;
};
