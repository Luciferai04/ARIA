#!/usr/bin/env node
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

const currentDir = process.cwd();
// Detect if running from npx cache (npm stores cached modules in _npx or similar)
const npxCacheTrigger = __dirname.includes('_npx') || __dirname.includes('npm-cache'); 

console.log("🚀 Starting ARIA (Agentic Research Intelligence Assistant) Pipeline...");

let projectDir = currentDir;

// If app.py is missing natively, it means they ran this command via npx in an empty directory.
// We should clone the repo to act like a scaffolding tool (`create-react-app` style).
if (!fs.existsSync(path.join(currentDir, 'app.py')) && npxCacheTrigger) {
    const defaultProjectName = process.argv[2] || 'aria-app';
    projectDir = path.join(currentDir, defaultProjectName);
    
    console.log(`📦 Creating new ARIA workspace in ./${defaultProjectName}...`);
    
    if (fs.existsSync(projectDir)) {
        console.error(`❌ Directory '${defaultProjectName}' already exists! Please delete it or navigate to an empty directory.`);
        process.exit(1);
    }
    
    try {
        execSync(`git clone https://github.com/soumyajitghosh/ARIA.git ${defaultProjectName}`, { stdio: 'inherit' });
    } catch (e) {
        console.error("❌ Failed to clone repository. Make sure Git is installed.");
        process.exit(1);
    }
} else if (!fs.existsSync(path.join(currentDir, 'app.py'))) {
    // If run directly but not from the root of the project
    projectDir = path.resolve(__dirname, '..');
}

process.chdir(projectDir);

console.log(`\n🐍 Verifying Python Environment...`);
let pythonCmd = 'python3';
try {
    execSync('python3 --version', { stdio: 'ignore' });
} catch (e) {
    // Fallback to python if python3 is not mapped natively
    pythonCmd = 'python';
}

if (!fs.existsSync(path.join(projectDir, '.venv'))) {
    console.log(`⚙️  Creating new sandboxed virtual environment (.venv)...`);
    execSync(`${pythonCmd} -m venv .venv`, { stdio: 'inherit' });
} else {
    console.log(`✅ Virtual environment already exists.`);
}

console.log(`📥 Checking and installing Python dependencies... (this might take a minute)`);
const pipCmd = process.platform === 'win32' ? path.join('.venv', 'Scripts', 'pip') : path.join('.venv', 'bin', 'pip');
const streamlitCmd = process.platform === 'win32' ? path.join('.venv', 'Scripts', 'streamlit') : path.join('.venv', 'bin', 'streamlit');

execSync(`${pipCmd} install -r requirements.txt`, { stdio: 'ignore' });

console.log(`\n🎉 Installation Perfect! Launching Streamlit Application natively...`);
console.log(`------------------------------------------------------------------\n`);

try {
    execSync(`${streamlitCmd} run app.py`, { stdio: 'inherit' });
} catch (e) {
    console.log("\n🛑 Successfully stopped ARIA.");
}
