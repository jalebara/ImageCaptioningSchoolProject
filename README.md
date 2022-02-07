# Project Name
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Turpis cursus in hac habitasse. Egestas erat imperdiet sed euismod. A cras semper auctor neque vitae tempus. Quis eleifend quam adipiscing vitae proin sagittis nisl rhoncus. Ante metus dictum at tempor commodo ullamcorper. Odio ut enim blandit volutpat maecenas volutpat blandit aliquam etiam. Tempor commodo ullamcorper a lacus vestibulum sed. Euismod lacinia at quis risus sed vulputate odio. Purus viverra accumsan in nisl nisi scelerisque eu ultrices vitae. Sed libero enim sed faucibus turpis in. Viverra suspendisse potenti nullam ac tortor vitae purus faucibus. Purus faucibus ornare suspendisse sed nisi. Dictum at tempor commodo ullamcorper a lacus vestibulum. Tincidunt lobortis feugiat vivamus at augue eget. Sed viverra tellus in hac habitasse. Sem viverra aliquet eget sit amet tellus cras adipiscing enim. Purus sit amet volutpat consequat mauris nunc congue. Volutpat consequat mauris nunc congue nisi vitae suscipit tellus.

# Basic Usage
1. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
2. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
3. Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.

## Example usage
```
"""The first step is to create an SMTP object, each object is used for connection 
with one server."""

import smtplib
server = smtplib.SMTP('smtp.gmail.com', 587)

#Next, log in to the server
server.login("youremailusername", "password")

#Send the mail
msg = "
Hello!" # The /n separates the message from the headers
server.sendmail("you@gmail.com", "target@example.com", msg)
```
## Video Demos

[Demo Video](https://www.youtube.com/watch?v=dQw4w9WgXcQ)