

package org.apache.tools.ant.taskdefs;

import org.apache.tools.ant.*;
import org.apache.tools.ant.types.*;

import java.util.Vector;
import java.io.File;
import java.io.IOException;

public class ExecuteOn extends ExecTask {

    protected Vector filesets = new Vector();
    private boolean parallel = false;
    protected String type = "file";
    protected Commandline.Marker srcFilePos = null;


    public void addFileset(FileSet set) {
        filesets.addElement(set);
    }


    public void setParallel(boolean parallel) {
        this.parallel = parallel;
    }


    public void setType(FileDirBoth type) {
        this.type = type.getValue();
    }


    public Commandline.Marker createSrcfile() {
        if (srcFilePos != null) {
            throw new BuildException(taskType + " doesn\'t support multiple srcfile elements.",
                                     location);
        }
        srcFilePos = cmdl.createMarker();
        return srcFilePos;
    }

    protected void checkConfiguration() {
        super.checkConfiguration();
        if (filesets.size() == 0) {
            throw new BuildException("no filesets specified", location);
        }
    }

    protected void runExec(Execute exe) throws BuildException {
        try {

            for (int i=0; i<filesets.size(); i++) {
                Vector v = new Vector();
                FileSet fs = (FileSet) filesets.elementAt(i);
                File base = fs.getDir(project);
                DirectoryScanner ds = fs.getDirectoryScanner(project);

                if (!"dir".equals(type)) {
                    String[] s = getFiles(base, ds);
                    for (int j=0; j<s.length; j++) {
                        v.addElement(s[j]);
                    }
                }

                if (!"file".equals(type)) {
                    String[] s = getDirs(base, ds);;
                    for (int j=0; j<s.length; j++) {
                        v.addElement(s[j]);
                    }
                }

                String[] s = new String[v.size()];
                v.copyInto(s);

                int err = -1;
                
                if (parallel) {
                    String[] command = getCommandline(s, base);
                    log("Executing " + Commandline.toString(command), 
                        Project.MSG_VERBOSE);
                    exe.setCommandline(command);
                    err = exe.execute();
                    if (err != 0) {
                        if (failOnError) {
                            throw new BuildException("Exec returned: "+err, 
                                                     location);
                        } else {
                            log("Result: " + err, Project.MSG_ERR);
                        }
                    }

                } else {
                    for (int j=0; j<s.length; j++) {
                        String[] command = getCommandline(s[j], base);
                        log("Executing " + Commandline.toString(command), 
                            Project.MSG_VERBOSE);
                        exe.setCommandline(command);
                        err = exe.execute();
                        if (err != 0) {
                            if (failOnError) {
                                throw new BuildException("Exec returned: "+err, 
                                                         location);
                            } else {
                                log("Result: " + err, Project.MSG_ERR);
                            }
                        }
                    }
                }
            }

        } catch (IOException e) {
            throw new BuildException("Execute failed: " + e, e, location);
        } finally {

            logFlush();
        }
    }


    protected String[] getCommandline(String[] srcFiles, File baseDir) {
        String[] orig = cmdl.getCommandline();
        String[] result = new String[orig.length+srcFiles.length];

        int index = orig.length;
        if (srcFilePos != null) {
            index = srcFilePos.getPosition();
        }
        System.arraycopy(orig, 0, result, 0, index);

        for (int i=0; i < srcFiles.length; i++) {
            result[index+i] = (new File(baseDir, srcFiles[i])).getAbsolutePath();
        }
        
        System.arraycopy(orig, index, result, index+srcFiles.length, 
                         orig.length-index);
        return result;
    }


    protected String[] getCommandline(String srcFile, File baseDir) {
        return getCommandline(new String[] {srcFile}, baseDir);
    }


    protected String[] getFiles(File basedir, DirectoryScanner ds) {
        return ds.getIncludedFiles();
    }


    protected String[] getDirs(File basedir, DirectoryScanner ds) {
        return ds.getIncludedDirectories();
    }


    public static class FileDirBoth extends EnumeratedAttribute {
        public String[] getValues() {
            return new String[] {"file", "dir", "both"};
        }
    }
}
