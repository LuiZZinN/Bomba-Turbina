const fs = require('fs');
let c = fs.readFileSync('app.py', 'utf8');

const t8_start = c.indexOf('# ==========================================\\n# TAB 8: Malha (Meshing)');
if (t8_start !== -1) {
    let t8_content = c.substring(t8_start);
    c = c.substring(0, t8_start);
    
    t8_content = t8_content.replace(/# ==========================================\n# TAB 8: Malha \(Meshing\)\n# ==========================================\nwith tab8:\n/, '');
    
    let lines = t8_content.split('\n');
    let inside_str = false;
    for (let i = 0; i < lines.length; i++) {
        if (lines[i].includes('"""')) {
            let count = (lines[i].match(/"""/g) || []).length;
            if (count % 2 !== 0) {
                if (!inside_str) {
                    lines[i] = '        ' + lines[i];
                    inside_str = true;
                } else {
                    inside_str = false;
                }
            } else {
                if (!inside_str && lines[i].trim() !== '') lines[i] = '        ' + lines[i];
            }
        } else {
            if (!inside_str && lines[i].trim() !== '') {
                lines[i] = '        ' + lines[i];
            }
        }
    }
    
    const insert_point = c.indexOf('    else:\n        st.info("Calcule a cinemática base primeiro.")\n\n# ==========================================\n# TAB 4: Validação CFD');
    
    if (insert_point !== -1) {
        c = c.substring(0, insert_point) + 
            '        st.markdown("---")\n' +
            lines.join('\n') + 
            '\n' +
            c.substring(insert_point);
        fs.writeFileSync('app.py', c);
        console.log('Successfully injected tab8 into tab3');
    } else {
        console.log('Failed to find insert point.');
    }
} else {
    console.log('Failed to find TAB 8.');
}
