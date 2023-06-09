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
package org.apache.camel.component.file;

import java.io.File;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;

/**
 * Unit test for the alwaysConsume=true option.
 */
public class FileAlwaysConsumeTest extends ContextTestSupport {

    @Override
    protected void setUp() throws Exception {
        super.setUp();
        deleteDirectory("target/alwaysconsume");
        template.sendBodyAndHeader("file://target/alwaysconsume/", "Hello World", FileComponent.HEADER_FILE_NAME, "report.txt");
    }

    @Override
    protected RouteBuilder createRouteBuilder() throws Exception {
        return new RouteBuilder() {
            public void configure() throws Exception {
                from("file://target/alwaysconsume/?consumer.alwaysConsume=true&moveNamePrefix=done/").to("mock:result");
            }
        };

    }

    public void testAlwaysConsume() throws Exception {
        // consume the file the first time
        MockEndpoint mock = getMockEndpoint("mock:result");
        mock.expectedBodiesReceived("Hello World");
        mock.expectedMessageCount(1);

        assertMockEndpointsSatisfied();

        Thread.sleep(1000);

        // reset mock and set new expectations
        mock.reset();
        mock.expectedBodiesReceived("Hello World");
        mock.expectedMessageCount(1);

        // move file back
        File file = new File("target/alwaysconsume/done/report.txt");
        File renamed = new File("target/alwaysconsume/report.txt");
        file = file.getAbsoluteFile();
        file.renameTo(renamed.getAbsoluteFile());

        // should consume the file again
        assertMockEndpointsSatisfied();
    }

}