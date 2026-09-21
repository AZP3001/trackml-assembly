// Content sources for the CSS build. Every file that can put a class name on
// an element has to be listed, including the JS: script.js builds button
// class strings at runtime (toggleRun, toggleHyper) and those literals are
// only discoverable by scanning it.
module.exports = {
    content: [
        './index.html',
        './script.js',
        './engine.js',
        './image-import.js',
        './tracks.js'
    ],
    theme: { extend: {} },
    plugins: []
};
