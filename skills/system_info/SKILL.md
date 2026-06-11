---
name: system_info
description: A skill that provides system information
---
# System Info
Use the System Info skill when a user asks for system information

## Workflow
1. User enters "system info"

``` js
const os = require('os');
console.log('Platform: ', os.platform());
console.log('Architecture: ', os.arch());
console.log('Hostname: ', os.hostname());
console.log('Total Memory: ', os.totalmem());
console.log('Free Memory: ', os.freemem());
console.log('Uptime: ', os.uptime());
console.log('Release: ', os.release());
console.log('Type: ', os.type());
```
