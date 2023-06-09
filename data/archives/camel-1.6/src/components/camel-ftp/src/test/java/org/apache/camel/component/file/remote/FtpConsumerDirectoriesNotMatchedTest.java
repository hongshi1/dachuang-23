/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.camel.component.file.remote;

import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.file.FileComponent;
import org.apache.camel.component.mock.MockEndpoint;

/**
 * Unit test that ftp consumer will not match directories (CAMEL-920)
 */
public class FtpConsumerDirectoriesNotMatchedTest extends FtpServerTestSupport {

    private int port = 20055;

    private String ftpUrl = "ftp://admin@localhost:" + port + "/dirnotmatched/?password=admin"
            + "&consumer.recursive=true&consumer.regexPattern=.*txt$";

    public void testSkipDirectories() throws Exception {
        MockEndpoint mock = getMockEndpoint("mock:result");
        mock.expectedMessageCount(3);
        mock.assertIsSatisfied();
    }

    public int getPort() {
        return port;
    }

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        prepareFtpServer();
    }

    private void prepareFtpServer() throws Exception {
        // prepares the FTP Server by creating files on the server that we want to unit
        // test that we can pool and store as a local file
        String ftpUrl = "ftp://admin@localhost:" + port + "/dirnotmatched";
        template.sendBodyAndHeader(ftpUrl + "/?password=admin", "This is a dot file",
                FileComponent.HEADER_FILE_NAME, ".skipme");
        template.sendBodyAndHeader(ftpUrl + "/?password=admin", "This is a web file",
                FileComponent.HEADER_FILE_NAME, "index.html");
        template.sendBodyAndHeader(ftpUrl + "/?password=admin", "This is a readme file",
                FileComponent.HEADER_FILE_NAME, "readme.txt");
        template.sendBodyAndHeader(ftpUrl + "/2007/?password=admin", "2007 report",
                FileComponent.HEADER_FILE_NAME, "report2007.txt");
        template.sendBodyAndHeader(ftpUrl + "/2008/?password=admin", "2008 report",
                FileComponent.HEADER_FILE_NAME, "report2008.txt");
    }

    protected RouteBuilder createRouteBuilder() throws Exception {
        return new RouteBuilder() {
            public void configure() throws Exception {
                from(ftpUrl).to("mock:result");
            }
        };
    }

}