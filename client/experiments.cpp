// TODO:
// - build request
// - send request
// - handle response

#include "base64.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <Poco/Exception.h>
#include <Poco/Net/HTTPClientSession.h>
#include <Poco/Net/HTTPRequest.h>
#include <Poco/Net/HTTPResponse.h>
#include <Poco/Path.h>
#include <Poco/StreamCopier.h>
#include <Poco/URI.h>

using namespace Poco::Net;
using namespace Poco;
using namespace std;

string ofPostRequest(string url, string body, map<string, string> headers) {
  try {
    // prepare session
    URI uri(url);
    HTTPClientSession session(uri.getHost(), uri.getPort());

    // prepare path
    string path(uri.getPathAndQuery());
    if (path.empty())
      path = "/";

    // send request
    HTTPRequest req(HTTPRequest::HTTP_POST, path, HTTPMessage::HTTP_1_1);
    req.setContentType("application/json");

    // Set headers here
    for (map<string, string>::iterator it = headers.begin();
         it != headers.end(); it++) {
      req.set(it->first, it->second);
    }

    // Set the request body
    req.setContentLength(body.length());

    // sends request, returns open stream
    std::ostream &os = session.sendRequest(req);
    os << body; // sends the body
    // req.write(std::cout); // print out request

    // get response
    HTTPResponse res;
    cout << res.getStatus() << " " << res.getReason() << endl;

    istream &is = session.receiveResponse(res);
    stringstream ss;
    StreamCopier::copyStream(is, ss);

    return ss.str();
  } catch (Exception &ex) {
    cerr << ex.displayText() << endl;
    return "";
  }
}

//--------------------------------------------------------------

struct Image {
  int id;
  string base64_img;
} image;

int main() {
  string line;
  string img64 = "";

  ifstream input("./test/000000007454.jpg", ios::in | ios::binary);

  if (input.is_open()) {

    while (getline(input, line)) {

      string encoded = base64_encode(
          reinterpret_cast<const unsigned char *>(line.c_str()), line.length());

      img64 += encoded;
    }

    input.close();
  }
  Image img;
  img.id = 2;
  img.base64_img = img64;

  map<string, string> headers;
  headers["Content-Type"] = "application/json";
  string body = "id=" + to_string(img.id) + "&image=" + img.base64_img;
  cout << ofPostRequest("http://localhost", body, headers);
}
