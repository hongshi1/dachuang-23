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
package org.apache.camel.component.http;

import org.apache.camel.ContextTestSupport;

/**
 * Unit test to query Google using GET with endpoint having the query parameters.
 */
public class HttpQueryGoogleTest extends ContextTestSupport {

    public boolean isUseRouteBuilder() {
        return false;
    }

    public void testQueryGoogle() throws Exception {
        Object out = template.sendBody("http://www.google.com/search?q=Camel", "");
        assertNotNull(out);
        String data = context.getTypeConverter().convertTo(String.class, out);
        assertTrue("Camel should be in search result from Google", data.indexOf("Camel") > -1);
    }
}