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
package org.apache.camel.component.jetty;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.component.mock.MockEndpoint;

/**
 * Unit test with a simple route test.
 */
public class JettyContentBasedRouteTest extends ContextTestSupport {

    private String serverUri = "http://localhost:5432/myservice";

    public void testSendOne() throws Exception {
        MockEndpoint mock = getMockEndpoint("mock:one");

        mock.expectedHeaderReceived("one", "true");

        template.sendBody(serverUri + "?one=true", null);

        assertMockEndpointsSatisfied();
    }

    public void testSendOther() throws Exception {
        MockEndpoint mock = getMockEndpoint("mock:other");

        mock.expectedHeaderReceived("two", "true");

        template.sendBody(serverUri + "?two=true", null);

        assertMockEndpointsSatisfied();
    }

    protected RouteBuilder createRouteBuilder() throws Exception {
        return new RouteBuilder() {
            public void configure() throws Exception {
                // START SNIPPET: e1
                from("jetty:" + serverUri)
                    .choice()
                    .when().simple("in.header.one").to("mock:one")
                    .otherwise()
                    .to("mock:other");
                // END SNIPPET: e1
            }
        };
    }

}