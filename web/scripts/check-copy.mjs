import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
function check(dir) { for(const entry of readdirSync(dir,{withFileTypes:true})) { const path=join(dir,entry.name); if(entry.isDirectory())check(path);else {const text=readFileSync(path,'utf8');if(/[\u2013\u2014]/.test(text))throw new Error(`Forbidden dash in ${path}`);} } }
check(new URL('../src',import.meta.url).pathname);console.log('Source dash gate: 0 em or en dashes');
