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
package org.apache.camel.processor;

import java.util.List;

import org.apache.camel.ContextTestSupport;
import org.apache.camel.Exchange;
import org.apache.camel.builder.RouteBuilder;
import org.apache.camel.builder.xml.Namespaces;
import org.apache.camel.component.mock.MockEndpoint;

/**
 * @version $Revision$
 */
public class SetHeaderTest extends ContextTestSupport {
    protected String matchingBody = "<person name='James' city='London'/>";

    public void testSendMatchingMessage() throws Exception {
        MockEndpoint resultEndpoint = getMockEndpoint("mock:result");
        resultEndpoint.expectedMessageCount(1);

        sendBody("direct:start", matchingBody);

        assertMockEndpointsSatisfied();
        List<Exchange> list = resultEndpoint.getReceivedExchanges();
        Exchange exchange = list.get(0);
        Object value = exchange.getIn().getHeader("foo");
        assertEquals("foo header", "London", value);
    }

    protected RouteBuilder createRouteBuilder() {
        return new RouteBuilder() {
            public void configure() {
                // START SNIPPET: example
                Namespaces ns = new Namespaces("foo", "urn:cheese");

                from("direct:start").
                        unmarshal().string().
                        setHeader("foo").xpath("/person[@name='James']/@city", String.class).
                        to("mock:result");
                // END SNIPPET: example
            }
        };
    }
}