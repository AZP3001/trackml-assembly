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
    // `wide:` is the side-panel layout: any screen 1280px+, plus anything in
    // landscape from 640px up — an iPad on its side or a phone held sideways
    // has the width for settings beside the track, just not the height to
    // stack them above it. styles.css carries the exact negation.
    theme: { extend: { screens: { wide: { raw: '(min-width: 1280px), (orientation: landscape) and (min-width: 640px)' } } } },
    plugins: []
};
